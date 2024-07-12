import os
from glob import glob
from pathlib import Path

import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # end of text token

data_dir = Path(os.path.dirname(__file__)) / "finewebedu_10B"


def ceil_div(x, y):
  return (x - 1) // y + 1


def get_dataloader(batch_size, context_len, split="train"):
  train_files = glob(str(data_dir / f"*{split}*.bin"))

  data_shards = [np.memmap(f, dtype=np.uint16, mode='r') for f in train_files]
  tokens_per_shard = data_shards[0].size
  tokens_per_mini_batch = ceil_div(16384, context_len) * context_len  # 32KB disk read

  def sample_a_mini_batch():
    shard = data_shards[np.random.randint(len(data_shards))]
    ix = np.random.randint(tokens_per_shard - tokens_per_mini_batch - 1)
    if ix >= shard.size:
      return None

    data = shard[ix:ix + tokens_per_mini_batch + 1]
    x_ = data[:-1]
    y_ = data[1:]
    # assert np.all(x_[1:] == y_[:-1])

    b = ceil_div(x_.size, context_len)
    x = np.full(shape=b * context_len, fill_value=eot)
    y = np.full(shape=b * context_len, fill_value=eot)
    x[:x_.size] = x_[:]
    y[:y_.size] = y_[:]
    return x.reshape((b, context_len)), y.reshape((b, context_len))

  def sample_batch():
    remain = batch_size
    x = []
    y = []
    while remain > 0:
      mini_batch = sample_a_mini_batch()
      if mini_batch is None:
        continue
      x_, y_ = mini_batch
      mb = x_.shape[0]
      # assert np.all(x_[:, 1:] == y_[:, :-1])
      x.append(x_[:min(mb, remain)])
      y.append(y_[:min(mb, remain)])
      remain -= mb

    return np.concatenate(x), np.concatenate(y)

  while True:
    yield sample_batch()


def get_batch(dl):
  return next(dl)


if __name__ == "__main__":
  np.random.seed(43)
  for i, (x, y) in enumerate(get_dataloader(480, 1024, split="val")):
    print(i, x.shape, y.shape)
    # assert np.all(x[:, 1:] == y[:, :-1])
    # if not np.all(x[:, 1:] == y[:, :-1]):
    #   print(x)
    #   print(y)
    #   np.save("x.npy", x)
    #   np.save("y.npy", y)
    #   break
