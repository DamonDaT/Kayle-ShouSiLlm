import math
import time
import torch
from param import ModelArgs
from model import Llama3

# load the dataset
with open('/home/dateng/code/tmp/minLlama3/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# the first 200 characters. It's just one continuous text document with all of the works of shakespeare back-to-back
print(text[:200])

params = ModelArgs()
tokenizer = params.tokenizer

# Train and test splits
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be our training dataset, the rest for validation
train_data = data[:n]
val_data = data[n:]

params = ModelArgs()
print(params)

model = Llama3(params, tokenizer).to(params.device)

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters()) / 1e3, 'K parameters')
print(model)


# data loading for training which generates a small batch of data of inputs x and targets y
def get_batch(split, batch_size):
    # whether we grab from our training or validation dataset
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - params.max_seq_len, (batch_size,))
    x = torch.stack([data[i:i + params.max_seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + params.max_seq_len + 1] for i in ix])
    x, y = x.to(params.device), y.to(params.device)
    return x, y


@torch.no_grad()
def estimate_loss(model, batch_size, eval_iters=5):  # to estimate loss during the training loop
    out = {}
    model.eval()  # sets model to eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size)
            logits, loss = model(X, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # just resets to training mode
    return out


# create a PyTorch optimizer
# this is not what they used, but this learning rate & weight decay work for our tiny minGemma
lr_init = 1e-2
weight_decay = 0.02
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=weight_decay)

# how long we want to train for
max_iters = 2000

# how often we want to check & see how our loss is doing
eval_interval = 100

# Warmup setup
warmup_iters = 50  # Number of warmup iterations
warmup_factor = 1e-3  # Warmup factor (initial learning rate is multiplied by this factor)

lr_final = 1e-5  # Minimum learning rate


def lr_lambda(current_iter):
    if current_iter < warmup_iters:
        # Warmup phase
        return warmup_factor + (1 - warmup_factor) * current_iter / warmup_iters
    else:
        # Cosine decay phase with minimum learning rate
        decay_iters = max_iters - warmup_iters
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (current_iter - warmup_iters) / decay_iters))
        return max(cosine_decay, lr_final / lr_init)


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

start_time = time.time()

# Enable anomaly detection. uncomment these lines if you need to do extensive debugging
# torch.autograd.set_detect_anomaly(True)

for iter in range(max_iters):

    # sample a batch of data
    xb, yb = get_batch('train', params.max_batch_size)

    # train
    logits, loss = model(xb, targets=yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Update the learning rate
    scheduler.step()

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        current_time = time.time()
        elapsed_time = current_time - start_time
        losses = estimate_loss(model, params.max_batch_size)
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"step {iter:04d}: lr {current_lr:.6f}, train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time elapsed: {elapsed_time:.2f} seconds")

# Disable anomaly detection after the training loop
# torch.autograd.set_detect_anomaly(False)
