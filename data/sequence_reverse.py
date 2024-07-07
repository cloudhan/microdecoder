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
    assert num_categories < 100
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
    sample_len = self.model_context_len + 1
    data = np.full(sample_len, fill_value=eog_id, dtype=np.uint8)
    mask = np.zeros(sample_len, dtype=np.uint8)  # mask for target

    l = self.np_rng.integers(self.max_seq_len)
    x = self.np_rng.integers(low=2, high=2 + self.num_categories, size=l, dtype=np.uint8)
    y = np.flip(x, axis=0)

    def fill_buffer(dst, src):
      if len(dst) < len(src):
        dst[:len(dst)] = src[:len(dst)]
      else:
        dst[:len(src)] = src

    fill_buffer(data, np.concatenate([x, _bog_token, y, _eog_token], axis=0))
    fill_buffer(mask, np.concatenate([np.zeros_like(x), np.array([0]), np.ones_like(y), np.array([1])], axis=0))
    return data, mask

  def __len__(self):
    # if self.streaming:
    #   raise NotImplementedError
    return self.size

  def __getitem__(self, idx):
    if self.streaming:
      return self._gen()
    else:
      return self.data[idx]

  def reset():
    pass


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
    assert isinstance(batch[0], tuple) and isinstance(batch[0][0], np.ndarray)
    data, mask = zip(*batch)
    return jnp.stack(data), jnp.stack(mask)

  dataset = load_dataset("sequence_reverse", split=split, streaming=True)
  return data.DataLoader(
      dataset[split],
      batch_size=batch_size,
      shuffle=True,
      collate_fn=collate,
  )


if __name__ == "__main__":
  dataloader = get_dataloader("train", 4)

  for idx, batch in enumerate(dataloader):
    if idx >= 2:
      break
    data, mask = batch
    print(idx, data.shape, mask.shape)
    print(idx, data, mask)
