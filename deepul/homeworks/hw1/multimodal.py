import numpy as np
from deepul.hw1_helper import (
    # Q1
    visualize_q1_data,
    q1_sample_data_1,
    q1_sample_data_2,
    q1_save_results,
    # Q2
    q2a_save_results,
    q2b_save_results,
    visualize_q2a_data,
    visualize_q2b_data,
    # Q3
    q3ab_save_results,
    q3c_save_results,
    # Q4
    q4a_save_results,
    q4b_save_results,
    # Q5
    visualize_q5_data,
    q5a_save_results,
    # Q6
    visualize_q6_data,
    q6a_save_results,
)

# from IPython.lib.deepreload import reload
# %load_ext autoreload
# %autoreload 2
from hw1 import *
from torch.utils.data import Dataset, DataLoader
import tiktoken as tk

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
            for c in text:
                if p not in self.token_dict:
                    raise Exception(f"Unseen character: {c}")
                else:
                    tokens.append(self.token_dict[c])
            tokens.append(self.token_dict['<eos>'])    
        return tokens

    def decode(self, tokens):
        inv_token_dict = {v: k for k, v in self.token_dict.items()}
        text = ''.join(inv_token_dict[t] for t in tokens)
        return text

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
from tiktoken._educational import *



def q5_a(train_text, test_text):
    """
    train_text: list[str] Train text sequences.
    test_text: list[str] Test text sequences.

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a list of 5 (str), 5 generated samples from the model.
    """
    batch_size = 128
    lr = 1e-3
    num_epochs = 200
    d_model = 128
    warmup_steps = 500
    num_batches = len(batch_indices(len(train_text), batch_size))
    total_steps = num_epochs * num_batches
    print(f'Total number of training steps: {total_steps}')
    data = "".join(train_text + test_text)
    enc = Tokenizer(data)
    # enc = tk.get_encoding('gpt2')
    # # Train a BPE tokeniser on a small amount of text
    # gpt2_pattern = (
    #     r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # )
    # with open(__file__, "r") as f:
    #     data = f.read()
    # 
    # enc = SimpleBytePairEncoding.train(data, vocab_size=600, pat_str=gpt2_pattern)
    # enc.n_vocab = 600
        
    config = GPTConfig()
    config.vocab_size = enc.n_vocab
    config.n_embd = d_model
    config.block_size = d_model
    model = TextGPT(config)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        # tokenize data
        train_dataset = TextDataset(enc.encode("".join(train_text)), block_size=d_model)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = TextDataset(enc.encode("".join(test_text)), block_size=d_model)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        num_batches = len(train_loader)
        train_losses = np.zeros((num_epochs, num_batches))
        test_losses = np.zeros(num_epochs + 1)

        # compute initial test loss
        test_loss, _ = q5_loop(test_loader, model)
        test_losses[0] = test_loss
        print(f'Initial test loss: {test_loss:.3f}')

    # define optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: lr_lambda(x, warmup_steps=warmup_steps, total_steps=total_steps))

    for epoch in range(num_epochs):
        train_loss, train_loss_batches = q5_loop(train_loader, model, optimizer)
        train_losses[epoch, :] = train_loss_batches
        test_loss, _ = q5_loop(test_loader, model)
        test_losses[epoch + 1] = test_loss
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}')
        # lr schedule update for epoch
        scheduler.step()

    # sample from model
    model.eval()
    # save model
    torch.save(model, f"text_model.pth")
    samp = "Thy love and mind,"
    start_tokens = enc.encode(samp)
    token_samples = model.sample(start_tokens, num_samples=5)
    text_samples = []
    for i, sample in enumerate(token_samples):
        token_sample = sample.detach().cpu().numpy()
        text_sample = enc.decode(token_sample)
        print(f'> {text_sample}')
        text_samples.append(text_sample)
    
    train_losses = train_losses.flatten()
    return train_losses, test_losses, text_samples



q5a_save_results(q5_a)