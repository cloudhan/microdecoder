import equinox as eqx
import functools
import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
from tqdm import tqdm

from model.gpt2 import GPT2, GPT2Config, GPT2_S, GPT2_M


def get_lr_schedule(base_lr, warmup_steps, train_steps):
  return optax.warmup_cosine_decay_schedule(
      init_value=1e-5 * base_lr,
      peak_value=base_lr,
      warmup_steps=warmup_steps,
      decay_steps=train_steps - warmup_steps,
  )


@eqx.filter_value_and_grad(allow_int=True)
def compute_loss(model, batch, key=None):
  input_ids, label_ids = batch
  logits = jax.vmap(model)(input_ids, key=jax.random.split(key, input_ids.shape[0]))
  losses = optax.losses.softmax_cross_entropy_with_integer_labels(logits, label_ids)
  return losses.mean()


@eqx.filter_jit
def train_step(model, batch, optimizer, opt_state, key=None):
  loss, grad = compute_loss(model, batch, key=key)
  updates, opt_state = optimizer.update(grad, opt_state, model)
  # model = optax.apply_updates(model, updates)
  model = eqx.apply_updates(model, updates)
  return loss, model, opt_state


def get_batch(batch_size, context_window_size):
  # reuse nanoGPT code for now
  data = np.memmap('train.bin', dtype=np.uint16, mode='r')
  ix = np.random.randint(len(data) - context_window_size, size=(batch_size,))
  x = np.stack([(data[i:i + context_window_size]).astype(np.int64) for i in ix])
  y = np.stack([(data[i + 1:i + 1 + context_window_size]).astype(np.int64) for i in ix])
  x, y = jnp.array(x), jnp.array(y)
  return x, y


def train(num_epochs, load_prefix=None, save_prefix=None):
  key = jax.random.PRNGKey(42)
  model = GPT2(GPT2_S, key=key)

  is_decayable = functools.partial(jax.tree_util.tree_map, lambda x: eqx.is_array(x) and x.ndim >= 2)
  optimizer = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(
          learning_rate=get_lr_schedule(6e-6 * 40 / 16 * 12, 2000, 19000),
          b1=0.9,
          b2=0.95,
          eps=1e-8,
          weight_decay=0.1,
          mask=is_decayable,
      ),
  )
  opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

  if load_prefix is not None:
    model = eqx.tree_deserialise_leaves(f"{load_prefix}.model.eqx", model)
    opt_state = eqx.tree_deserialise_leaves(f"{load_prefix}.opt_state.eqx", opt_state)
  model = eqx.nn.inference_mode(model, False)

  batch_size = 16
  context_window_size = 1024
  num_tokens_per_batch = batch_size * context_window_size
  tokens_trained_on = 0

  total_step = 0
  for epoch in tqdm(range(1, num_epochs + 1)):
    loss_acc = 0
    step = 0
    while step < 512:
      step_start_time = time.time()
      step += 1
      batch = get_batch(batch_size, context_window_size)
    #   break
    # while True:
      key, train_step_key = jax.random.split(key)
      loss, model, opt_state = train_step(model, batch, optimizer, opt_state, key=train_step_key)
      loss_acc += loss
      total_step += 1
      step_end_time = time.time()
      duration_ms = (step_end_time - step_start_time) * 1000
      tokens_trained_on += num_tokens_per_batch
      print(f"Epoch[{epoch}] | step:{step:5d} | step time:{duration_ms:4.2f}ms | "
            f"tokens:{tokens_trained_on*1e-9:6.4f}B | loss:{loss:6.4f}")

    if save_prefix is not None:
      eqx.tree_serialise_leaves(f"{save_prefix}.{total_step:06d}.model.eqx", model)
      eqx.tree_serialise_leaves(f"{save_prefix}.{total_step:06d}.opt_state.eqx", opt_state)

    print(f"Epoch[{epoch}] | step:{total_step} | loss acc:{loss_acc}")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--load_prefix", default=None, type=str)
  parser.add_argument("--save_prefix", default=None, type=str)
  # parser.add_argument("--batch_size", default=256, type=int)
  parser.add_argument("--num_epochs", default=25, type=int)
  args = parser.parse_args()

  train(args.num_epochs, args.load_prefix, args.save_prefix)
