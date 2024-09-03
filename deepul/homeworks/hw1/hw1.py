from dataclasses import dataclass
from tqdm import tqdm
from functools import partial
import copy
import math
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
# set device
import os
# os.environ['CUDA_LAUNCH_BLOCKING']="1"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def batch_indices(n, batch_size):
    """
    Generator of indices for each batch

    Args:
        n: number of samples in dataset
    """
    indices = np.arange(n)
    np.random.shuffle(indices)
    batches = []
    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        batches.append(torch.from_numpy(indices[start_idx:end_idx]))
    return batches


def q1_nll(theta, x):
    return torch.logsumexp(theta, dim=0) - theta[x].mean()


def create_batches(data, batch_size):
    np.random.shuffle(data)
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:i+batch_size])
    return batches


def pixel_to_token(pixels, C, K):
    """
    Vectorized conversion of pixels to tokens using PyTorch.
    Each unique vector of channel value [r, g, b]
    where {r, g, b} \in [0, ..., K-1] is converted to a single token
    There are K^C possible discrete token values

    Args:
        pixels: A tensor of pixels with shape (B*H*W, C).
        K: The number of possible discrete values for each pixel channel.
        C: The number of channels.
    Returns:
        tokens: torch.tensor, (B*H*W, 1), w/ possible values in [0, K^C - 1]
    """
    # Create a multiplier for each channel's contribution to the token.
    # This will be [K^(C-1), K^(C-2), ..., K^0] for C channels.
    multipliers = K ** torch.arange(C - 1, -1, -1,
                                    device=pixels.device, dtype=pixels.dtype)

    # Multiply each pixel by its channel multiplier and sum to get the token.
    tokens = torch.sum(pixels * multipliers, dim=1)
    return tokens


def token_to_pixel(tokens, C, K):
    """
    Vectorized mapping of tokens back to their original RGB (or single channel) values.

    Args:
        tokens: torch.tensor, shape (B*H*W,), tokens representing the pixels.
        C: Integer, number of color channels.
        K: Integer, number of possible discrete values for each pixel/channel.

    Returns:
        pixels: torch.tensor, shape (B*H*W, C), the original pixel values.
    """
    # Prepare the divisor for each channel based on its position
    divisors = K ** torch.arange(C-1, -1, -1,
                                 device=tokens.device, dtype=tokens.dtype)
    # Compute pixel values for each channel
    pixels = tokens.unsqueeze(1) // divisors % K
    return pixels


def tokenize_images(data, K):
    """
    Tokenize an array of images (vectorized)

    Args:
        data: (B, H, W, C)
        K: possible discrete values
    """
    # flatten the spatial dimensions and keep the color channel last
    B, H, W, C = data.shape
    flattened_images = data.reshape(-1, C)  # (B * H * W, C)

    tokens = pixel_to_token(flattened_images, C, K)

    # reshape back to the original batch of images
    # with spatial dimensions flattened
    tokens = tokens.view(B, H * W)

    # add special token K**C to represent <bos>
    # to the beginning of each image
    bos_token = torch.full((B, 1), K**C, dtype=torch.long, device=data.device)
    tokens = torch.cat([bos_token, tokens], dim=1)  # (B, H * W + 1)
    return tokens


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    Attention module for GPT model hw1
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.kv_attn = nn.Linear(
            config.n_embd, 2 * config.n_embd, bias=config.bias)
        self.q_attn = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.cache = None if config.cache is False else {'K': None, 'V': None}
        # If flash attention is available, use it
        self.flash = hasattr(torch.nn.functional,
                             'scaled_dot_product_attention')
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, use_cache=False):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        if use_cache:
            k, v = self.kv_attn(x[:, -1:, :]).split(self.n_embd, dim=2)
            q = self.q_attn(x)
            if self.cache['K'] is None:
                self.cache['K'] = k
                self.cache['V'] = v
            else:
                k = torch.cat([self.cache["K"], k], dim=1)
                v = torch.cat([self.cache["V"], v], dim=1)
                self.cache['K'] = k
                self.cache['V'] = v
        else:
            k, v = self.kv_attn(x).split(self.n_embd, dim=2)
            q = self.q_attn(x)

        k = k.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            # Causal Attention Mask
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

    @property
    def reset_cache(self):
        self.cache = {'K': None, 'V': None}


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(
            config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPTBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, use_cache=False):
        x = x + self.attn(self.ln_1(x), use_cache)
        x = x + self.mlp(self.ln_2(x))
        return x

    def reset_cache(self):
        self.attn.reset_cache


@dataclass
class GPTConfig:
    block_size: int = 1024
    # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 768
    dropout: float = 0.5
    # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    bias: bool = True
    cache: bool = True


@dataclass
class PIXELConfigBW:
    block_size: int = 1024
    vocab_size: int = 2**2
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True
    cache: bool = True


@dataclass
class PIXELConfigRGB:
    block_size: int = 1024
    vocab_size: int = 3**4
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False
    cache: bool = True


@dataclass
class PIXELConfigVQVAE:
    block_size: int = 1024
    vocab_size: int = 2**2
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True
    cache: bool = True


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # token embedding
            wte=nn.Embedding(config.vocab_size + 1, config.n_embd),
            # positional encoding/embedding
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, use_cache=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        # pos = pos.to(device)
        # forward the GPT model itself
        # token embeddings of shape (b, t, n_embd)
        tok_emb = self.transformer.wte(idx)
        # position embeddings of shape (t, n_embd)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, use_cache)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits[:, :-1, :]
            loss = F.cross_entropy(logits.transpose(
                1, 2), targets, ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # note: using list [-1] to preserve the time dim
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def reset_cache(self):
        for block in self.transformer.h:
            block.reset_cache()

    def sample(self, image_shape, K, num_samples=100, return_time=False, use_cache=True):
        B = num_samples
        H, W, C = image_shape
        # time for each sampling step over the length of the sequence
        sampling_times = np.zeros(H*W)

        # generate samples in batch
        with torch.no_grad():
            start_time = time.time()
            # start with <bos> token (vocab_size)
            bos = K**C
            x = torch.full((B, 1), bos, device=device)
            self.reset_cache()
            # iterate over each pixel
            for pixel_index in range(H * W):
                start_time = time.time()
                logits, loss = self.forward(
                    x, use_cache=use_cache)  # x: (B, curr_len)
                # use logits = (B, vocab_size) to sample a single token
                probs = F.softmax(logits.squeeze(), dim=-1)  # (B, vocab_size)
                next_token = torch.multinomial(probs, 1)  # (B, 1)
                x = torch.cat((x, next_token), dim=-1)  # (B, curr_len + 1)
                sampling_times[pixel_index] = time.time() - start_time

            # covert token back to pixel value to store in sample tensor
            # x: (B*H*W,), ignoring bos
            x = x[:, 1:].flatten()  # (B*H*W,)
            samples = token_to_pixel(x, C, K).view(B, H, W, C)

        samples = samples.detach().cpu().numpy()

        if return_time:
            return samples, sampling_times
        else:
            return samples

class TextGPT(GPT):
    
    def __init__(self, config):
        super().__init__(config)

    def sample(self, sequence, num_samples=5, sentance_size = 128, use_cache=False):

        samples = np.zeros((num_samples, sentance_size))

        with torch.no_grad():
            self.reset_cache()
            # for i in tqdm(range(num_samples)):
            # initialize quantized image with bos
            #x = torch.full((num_samples, 1), 85, device=device)
            x = torch.tensor(sequence, device=device).unsqueeze(0).repeat(num_samples, 1)
            # iterate over each pixel
            while x.size(1) < sentance_size:
                logits, loss = self.forward(x, use_cache=use_cache)
                # use logits = (1, vocab_size) to sample a single token
                probs = F.softmax(logits.squeeze(), dim=-1)  # (B, vocab_size)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # (1, 1)
                ix = torch.multinomial(topk_probs, 1)
                # (1, current_len)
                xcol = topk_indices.gather(-1, ix)
                
                x = torch.cat((x, xcol), dim=-1)

            # ignore bos, reshape to (7, 7)
            samples = x
        return samples

def q3_loop(
        data, model, batch_size, loss_fn, optimizer=None):
    """
    Training or evaluation loop over all batches
    """
    if optimizer:
        model.train()
    else:
        model.eval()

    batches = batch_indices(data.shape[0], batch_size)
    loss_total = 0
    num_batches = len(batches)
    loss_batches = np.zeros(num_batches)

    for i, batch_idx in enumerate(tqdm(batches)):
        if optimizer:
            optimizer.zero_grad()

        batch = data[batch_idx]
        labels = batch[:, 1:]

        logits, loss = model(batch, labels)
        loss_total += loss.item() * batch_idx.shape[0]
        loss_batches[i] = loss.item()

        if optimizer:
            loss.backward()
            optimizer.step()

    loss_total /= data.shape[0]
    return loss_total, loss_batches


def lr_lambda(current_step, warmup_steps, total_steps):
    if current_step < warmup_steps:
        return current_step / warmup_steps
    else:
        # cosine decay
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))


class VQVAEDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        """
        Args:
            dataset: (N, 7, 7) int numpy array of vqvae qunatized images
                values in [0, 1024)
        """
        super().__init__()
        self.dataset = torch.from_numpy(dataset).to(device)

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return self.dataset[idx]  # (50,)


def vqvae_quantize_dataset(vqvae, images):
    """
    Quantize images with VQVAE in batches then combine

    Args:
        vqvae: trained VQVAE
        images: (N, H, W, C)
    """
    batch_size = 2056
    num_images = images.shape[0]
    quantized_images = []
    for start_idx in range(0, num_images, batch_size):
        end_idx = min(start_idx + batch_size, num_images)
        batch = images[start_idx:end_idx]  # (B, H, W, C)
        quantized_batch = vqvae.quantize(batch).reshape(
            (batch.shape[0], -1))  # (B, 49)
        bos = np.full((batch.shape[0], 1),
                      vqvae.n_embeddings, dtype=int)  # (B, 1)
        quantized_batch = np.concatenate(
            [bos, quantized_batch], axis=1)  # (B, 50)
        quantized_images.append(quantized_batch)
    quantized_images = np.concatenate(quantized_images)  # (N, 50)
    return quantized_images


class GPT_VQVAE(GPT):
    """
    Implements autogressive transformer with VAVAE sampling
    """

    def __init__(self, config):
        super().__init__(config)

    def sample(self, vqvae, image_shape, num_samples=100, use_cache=True):
        H, W, C = image_shape
        samples = np.zeros((num_samples, H, W, C))

        with torch.no_grad():
            self.reset_cache()
            # for i in tqdm(range(num_samples)):
            # initialize quantized image with bos
            bos = vqvae.n_embeddings
            x = torch.full((num_samples, 1), bos, device=device)
            # iterate over each pixel
            for _ in range(7 * 7):
                logits, loss = self.forward(x, use_cache=use_cache)
                # use logits = (1, vocab_size) to sample a single token
                probs = F.softmax(logits.squeeze(), dim=-1)  # (B, vocab_size)
                # (1, 1)
                next_token = torch.multinomial(probs, 1)
                # (1, current_len)
                x = torch.cat((x, next_token), dim=-1)

            # ignore bos, reshape to (7, 7)
            x = x[:, 1:].reshape(-1, 7, 7)
            # covert quantized image back to image to store in sample tensor
            image = vqvae.decode(x.detach().cpu().numpy())
            samples = image

        return samples


def q4_loop(
        loader, model, optimizer=None):
    """
    Training or evaluation loop over all batches
    Pass in a dataloader
    """
    if optimizer:
        model.train()
    else:
        model.eval()

    loss_total = 0
    loss_batches = np.zeros(len(loader))

    for i, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)
        if optimizer:
            optimizer.zero_grad()

        # shift target by one so the model is predicting the next token
        labels = batch[:, 1:]
        _, loss = model(batch, labels)
        # sum of loss of each sample in batch
        loss_total += loss.item() * batch.shape[0]
        loss_batches[i] = loss.item()

        if optimizer:
            loss.backward()
            optimizer.step()

    # mean loss over all samples in the dataset
    loss_total /= len(loader.dataset)
    return loss_total, loss_batches



class Tokenizer:
    def __init__(self, data):
        self.token_dict = self.create_token_dict(data)
        self.n_vocab = len(self.token_dict)
                           
    def create_token_dict(self, data):
        token_dict = {k:v for v,k in enumerate(sorted(set(data)))}
        token_dict['<bos>']=len(token_dict)
        token_dict['<eos>']=len(token_dict)
        return token_dict

    def __len__(self):
        return len(self.token_dict)

    def encode(self, text):
        tokens = []
        for p in text:
            tokens.append(self.token_dict['<bos>'])
            for c in p:
                if c not in self.token_dict:
                    raise Exception(f"Unseen character: {c}")
                else:
                    tokens.append(self.token_dict[c])
            tokens.append(self.token_dict['<eos>'])    
        return tokens
    
    def pencode(self, text, join=False):
        tokens = [[self.token_dict[c] for c in p] for p in text]
        return tokens
    
    def decode(self, tokens):
        inv_token_dict = {v: k for k, v in self.token_dict.items()}
        text = ''.join(inv_token_dict[t] for t in tokens)
        return text

class WordTokenizer:
    def __init__(self, data, start=0):
        self.token_dict = self.create_token_dict(data, start=start)
        self.inv_token_dict = {v: k for k, v in self.token_dict.items()}
        self.n_vocab = len(self.token_dict)
                           
    def create_token_dict(self, data, start=0):
        words = [w for d in data for w in d.split(" ")]
        #flatten list of lists
        token_dict = { k:(start+v) for v, k in enumerate(sorted(set(words)))}
        token_dict['<bos>']=start+len(token_dict)
        token_dict['<eos>']=start+len(token_dict)
        token_dict['<eoi>']=start+len(token_dict)
        return token_dict

    def __len__(self):
        return len(self.token_dict)

    def encode(self, text):
        token_list = []
        for p in text:
            tokens = [self.token_dict['<bos>']]
            tokens += [self.token_dict[w] for w in p.split(" ")]
            token_list.append(tokens)
        return token_list
        
    def decode(self, tokens):
        
        tokens = tokens
        text_list = []
        for p in tokens:
            text = [self.inv_token_dict[t] for t in p]
            text = ' '.join(text)
            text_list.append(text)
        return text_list

class VQVAEQuantizer:
    def __init__(self, vqvae):
        self.vqvae = vqvae
        self.n_embeddings = vqvae.n_embeddings
        self.batch_size = 2056
        
    def encode(self, images):
        """
        Quantize images with VQVAE in batches then combine

        Args:
            vqvae: trained VQVAE
            images: (N, H, W, C)
        """
        batch_size = 2056
        num_images = images.shape[0]
        quantized_images = []
        for start_idx in range(0, num_images, batch_size):
            end_idx = min(start_idx + batch_size, num_images)
            batch = images[start_idx:end_idx]  # (B, H, W, C)
            quantized_batch = self.vqvae.quantize(batch).reshape(
                (batch.shape[0], -1))  # (B, 49)
            bos = np.full((batch.shape[0], 1),
                      self.n_embeddings, dtype=int)  # (B, 1)
            quantized_batch = np.concatenate(
                [bos, quantized_batch], axis=1)  # (B, 50)
            quantized_images.append(quantized_batch)
        quantized_images = np.concatenate(quantized_images)  # (N, 50)
        return quantized_images


    
class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.block_size = block_size
        self.data = data
        self.samples = self._chunk_tokens()
        self.num_batches = len(self.samples)


    def __len__(self):
        return self.num_batches

    def _chunk_tokens(self):
        # Chunk data into blocks of block_size
        n_chunks = len(self.data) // self.block_size
        samples = self.data[:n_chunks * self.block_size] 
        samples += self.data[-self.block_size:]
        return torch.tensor(samples).view(-1, self.block_size)

    def __getitem__(self, idx):
        return self.samples[idx]

def q5_loop(
        loader, model, optimizer=None):
    """
    Training or evaluation loop over all batches
    Pass in a dataloader
    """
    if optimizer:
        model.train()
    else:
        model.eval()

    loss_total = 0
    loss_batches = np.zeros(len(loader))

    for i, data in enumerate(tqdm(loader)):
        batch = data
        batch = batch.to(device)
        if optimizer:
            optimizer.zero_grad()

        # shift target by one so the model is predicting the next token
        labels = batch[:, 1:]
        _, loss = model(batch, labels)
        # sum of loss of each sample in batch
        loss_total += loss.item() * batch.shape[0]
        loss_batches[i] = loss.item()

        if optimizer:
            loss.backward()
            optimizer.step()

    # mean loss over all samples in the dataset
    loss_total /= len(loader)
    return loss_total, loss_batches


class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.block_size = block_size
        self.data = data
        self.samples = self._chunk_tokens()
        self.num_batches = len(self.samples)


    def __len__(self):
        return self.num_batches

    def _chunk_tokens(self):
        # Chunk data into blocks of block_size
        n_chunks = len(self.data) // self.block_size
        samples = self.data[:n_chunks * self.block_size] 
        samples += self.data[-self.block_size:]
        return torch.tensor(samples).view(-1, self.block_size)

    def __getitem__(self, idx):
        return self.samples[idx]

def q5_loop(
        loader, model, optimizer=None):
    """
    Training or evaluation loop over all batches
    Pass in a dataloader
    """
    if optimizer:
        model.train()
    else:
        model.eval()

    loss_total = 0
    loss_batches = np.zeros(len(loader))

    for i, data in enumerate(tqdm(loader)):
        batch = data
        batch = batch.to(device)
        if optimizer:
            optimizer.zero_grad()

        # shift target by one so the model is predicting the next token
        labels = batch[:, 1:]
        _, loss = model(batch, labels)
        # sum of loss of each sample in batch
        loss_total += loss.item() * batch.shape[0]
        loss_batches[i] = loss.item()

        if optimizer:
            loss.backward()
            optimizer.step()

    # mean loss over all samples in the dataset
    loss_total /= len(loader)
    return loss_total, loss_batches



class MMGPT(GPT):
    """
    Implements autogressive MULTIMODAL transformer with VAVAE sampling
    """

    def __init__(self, config):
        super().__init__(config)

    def sample_image(self, sequence, vqvae, image_shape):
        H, W, C = image_shape
        seq = torch.tensor(sequence, device=device)
        samples = np.zeros((len(sequence), H, W, C))
        n_seq = seq.shape[1]
        with torch.no_grad():
            self.reset_cache()
            # for i in tqdm(range(num_samples)):
            # initialize quantized image with bos
            bos = torch.tensor(np.full((samples.shape[0], 1),
                      vqvae.n_embeddings, dtype=int), device=device)  # (B, 1)
            x = torch.concat([seq,bos], axis=1)
            # iterate over each pixel
            img = torch.zeros((samples.shape[0], H, W, C))
            for _ in range(7 * 7):
                logits, loss = self.forward(x)
                logits = logits[:, -1, :vqvae.n_embeddings]
                # use logits = (1, vocab_size) to sample a single token
                probs = F.softmax(logits.squeeze(), dim=-1)  # (B, vocab_size)
                # (1, 1)
                next_token = torch.multinomial(probs, 1)
                # (1, current_len)
                x = torch.cat((x, next_token), dim=-1)

            # ignore bos, reshape to (7, 7)
            x = x[:, n_seq+1:].reshape(-1, 7, 7)
            # covert quantized image back to image to store in sample tensor
            image = vqvae.decode(x.detach().cpu().numpy())
            samples = image

        return samples

    def sample_text(self, sequence, sentance_size = 7):
        n_samples = len(sequence)
        samples = np.zeros((n_samples, sentance_size))
        with torch.no_grad():
            self.reset_cache()
            # for i in tqdm(range(num_samples)):
            # initialize quantized image with bos
            #x = torch.full((num_samples, 1), 85, device=device)
            bos = torch.tensor(np.full((samples.shape[0], 1),
                      1024+1, dtype=int), device=device)  # (B, 1)
            x = torch.tensor(sequence, device=device)
            x = torch.concat([x, bos], axis=1)
            for _ in range(sentance_size):
                logits, loss = self.forward(x)
                logits = logits[:,:,1025:]
                # use logits = (1, vocab_size) to sample a single token
                probs = F.softmax(logits.squeeze(), dim=-1)  # (B, vocab_size)
                # (1, 1)
                next_token = torch.multinomial(probs, 1)+1025
                # (1, current_len)
                x = torch.cat((x, next_token), dim=-1)

            samples = x[:, -sentance_size:-1]
        return samples


class MultimodalDataset(Dataset):
    def __init__(self, img_data, text_data, vq, wt, rando_p = 0.5):
        super().__init__()
        self.vqvae_quantizer = vq
        self.word_tokenizer = wt
        self.token_dict = self.word_tokenizer.token_dict
        self.text_data = torch.tensor(self.word_tokenizer.encode(text_data))
        self.img_data = torch.tensor(self.vqvae_quantizer.encode(img_data))
        self.eos = torch.tensor([self.token_dict['<eos>']])
        self.eoi = torch.tensor([self.token_dict['<eoi>']])
        self.random_p = rando_p 

    def __len__(self):
        return self.img_data.shape[0]

    def __getitem__(self, idx):
        img_sample = self.img_data[idx]
        text_sample = self.text_data[idx]
        if random.random() > self.random_p:
            return torch.concat([self.eoi, text_sample, self.eos, img_sample])
        return torch.concat([self.eos, img_sample, self.eoi, text_sample])
    
    