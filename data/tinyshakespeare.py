import os
from pathlib import Path

import numpy as np

data_dir = Path(os.path.dirname(__file__)) / "tinyshakespeare"

def get_dataloader(batch_size, context_len, split="train"):
  data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode='r')
  while True:
    ix = np.random.randint(len(data) - context_len, size=(batch_size,))
    x = np.stack([(data[i:i + context_len]) for i in ix])
    y = np.stack([(data[i + 1:i + 1 + context_len]) for i in ix])
    yield x, y

def get_batch(dl):
  return next(dl)

if __name__ == "__main__":
  dl = get_dataloader(480, 1024)
  x, y = get_batch(dl)
  print(x.shape, y.shape)
  assert np.all(x[:, 1:] == y[:, :-1])
