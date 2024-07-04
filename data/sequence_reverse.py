import numpy as np
import jax.numpy as jnp
import torch.utils.data as data
from datasets import DatasetDict

bog_id = 0  # begin of generation
eog_id = 1  # end of generation
_bog_token = np.array([bog_id])
_eog_token = np.array([eog_id])


def ceil_div(x, y):
  return (x - 1) // y + 1


def get_model_context_len(max_seq_len):
  return ceil_div(max_seq_len * 2 + 2, 8) * 8


class ReverseDataset(data.Dataset):

  def __init__(self, num_categories, max_seq_len, size, np_rng, streaming=False):
    super().__init__()
    self.num_categories = num_categories
    self.max_seq_len = max_seq_len
    self.model_context_len = get_model_context_len(max_seq_len)
    self.size = size
    self.np_rng = np_rng
    self.streaming = streaming

    self.data = None
    if not self.streaming:
      data = []
      for _ in range(self.size):
        data.append(self._gen())

      self.data = np.stack(data, axis=0)

  def _gen(self):
    sample_len = self.model_context_len
    remaining_gen_seq_len = sample_len
    sample = []
    while remaining_gen_seq_len > 0:
      l = self.np_rng.integers(self.max_seq_len)
      x = self.np_rng.integers(low=2, high=2 + self.num_categories, size=l)
      y = np.flip(x, axis=0)
      sample.extend([x, _bog_token, y, _eog_token])
      remaining_gen_seq_len -= l * 2 + 2
    return np.concatenate(sample, axis=0)[:sample_len]

  def __len__(self):
    # if self.streaming:
    #   raise NotImplementedError
    return self.size

  def __getitem__(self, idx):
    if self.streaming:
      return self._gen()
    else:
      return self.data[idx]


def load_dataset(path, split=None, streaming=False):
  num_categories = 10
  max_seq_len = 64
  assert path == "sequence_reverse"
  if split == "train":
    return DatasetDict({
        "train": ReverseDataset(num_categories, max_seq_len, 65536, np.random.default_rng(42), streaming=streaming),
    })
  elif split == "test_val":
    return DatasetDict({
        "test_val": ReverseDataset(num_categories, max_seq_len, 4096, np.random.default_rng(43), streaming=streaming),
    })
  else:
    return DatasetDict({
        "train": ReverseDataset(num_categories, max_seq_len, 65536, np.random.default_rng(42), streaming=streaming),
        "test_val": ReverseDataset(num_categories, max_seq_len, 4096, np.random.default_rng(43), streaming=streaming),
    })


def get_dataloader(split, batch_size):

  def collate(batch):
    assert isinstance(batch[0], np.ndarray)
    return jnp.stack(batch)

  dataset = load_dataset("sequence_reverse", split=split, streaming=True)
  return data.DataLoader(
      dataset[split],
      batch_size=batch_size,
      shuffle=True,
      collate_fn=collate,
  )

if __name__ == "__main__":
  dataloader = get_dataloader("train", 64)

  for idx, batch in enumerate(dataloader):
    if idx >= 2:
      break
    print(idx, batch.shape, type(batch))
