import equinox as eqx
import functools
import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
from tqdm import tqdm

from model.gpt2 import GPT2, GPT2Config, GPT2_S, GPT2_M

batch_size = 480
mini_batch_size = 12
context_len = 1024
num_epochs = 25

mini_steps = batch_size // mini_batch_size
assert batch_size % mini_batch_size == 0


def get_lr_schedule(base_lr, min_lr, warmup_steps, train_steps):
  return optax.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=base_lr,
      warmup_steps=warmup_steps,
      decay_steps=train_steps - warmup_steps,
      end_value=min_lr,
  )


@eqx.filter_value_and_grad(allow_int=True)
def compute_loss(model, batch, key=None):
  input_ids, label_ids = batch
  logits = jax.vmap(model)(input_ids, key=jax.random.split(key, input_ids.shape[0]))
  losses = optax.losses.softmax_cross_entropy_with_integer_labels(logits, label_ids)
  return losses.mean()


@eqx.filter_jit
def train_step(model, mini_batches, optimizer, opt_state, key=None):

  def mini_step(carry, mini_batch):
    loss, grad, key = carry
    with jax.named_scope(f"minibatch_fwd"):
      key, step_key = jax.random.split(key, 2)
      step_loss, step_grad = compute_loss(model, mini_batch, key=step_key)
    with jax.named_scope(f"minibatch_bwd"):
      loss += step_loss
      grad = jax.tree_util.tree_map(jnp.add, grad, step_grad)
    return (loss, grad, key), None

  # scan over the mini_batches, each step work a slice of mini_batches[i], first dim
  loss = jnp.zeros(())
  grad = jax.tree_util.tree_map(jnp.zeros_like, eqx.filter(model, eqx.is_array))
  (loss, grad, _), _ = jax.lax.scan(mini_step, (loss, grad, key), mini_batches)

  # mini_batch_size = mini_batches[0]
  # batch_size = mini_batch_size * mini_steps
  # scale = mini_batch_size / batch_size, aka 1/mini_steps
  scale = 1.0 / mini_steps
  loss *= scale
  grad = jax.tree_util.tree_map(lambda x: scale * x, grad)

  updates, opt_state = optimizer.update(grad, opt_state, model)
  # model = optax.apply_updates(model, updates)
  model = eqx.apply_updates(model, updates)
  return loss, model, opt_state


def get_batch(batch_size, context_len):
  # reuse nanoGPT code for now
  data = np.memmap('train.bin', dtype=np.uint16, mode='r')
  ix = np.random.randint(len(data) - context_len, size=(batch_size,))
  x = np.stack([(data[i:i + context_len]).astype(np.int64) for i in ix])
  y = np.stack([(data[i + 1:i + 1 + context_len]).astype(np.int64) for i in ix])
  x, y = jnp.array(x), jnp.array(y)
  return x, y


def get_mini_batches(mini_batch_size, mini_steps, context_len):
  batch = get_batch(mini_batch_size * mini_steps, context_len)
  return (
      batch[0].reshape(mini_steps, mini_batch_size, context_len),
      batch[1].reshape(mini_steps, mini_batch_size, context_len),
  )

def count_params(model):
  weights = eqx.filter(model, eqx.is_array)
  return sum(x.size for x in jax.tree_util.tree_leaves(weights))


def train(num_epochs, load_prefix=None, save_prefix=None):
  key = jax.random.PRNGKey(42)
  model = GPT2(GPT2_S, key=key)
  param_count = count_params(model)
  print(f"parameters: {param_count * 1e-6:6.2f}M ({param_count})")

  is_decayable = functools.partial(jax.tree_util.tree_map, lambda x: eqx.is_array(x) and x.ndim >= 2)
  optimizer = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(
          learning_rate=get_lr_schedule(6e-4, 6e-5, 2000, 19000),
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

  num_tokens_per_batch = batch_size * context_len
  tokens_trained_on = 0

  total_step = 0
  for epoch in tqdm(range(1, num_epochs + 1)):
    loss_acc = 0
    step = 0
    while step < 256:
      step_start_time = time.time()
      step += 1
      mini_batches = get_mini_batches(mini_batch_size, mini_steps, context_len)
    #   break
    # while True:
      key, train_step_key = jax.random.split(key)
      loss, model, opt_state = train_step(model, mini_batches, optimizer, opt_state, key=train_step_key)
      loss_acc += loss
      total_step += 1
      step_end_time = time.time()
      duration_ms = (step_end_time - step_start_time) * 1000
      tokens_trained_on += num_tokens_per_batch
      print(f"Epoch[{epoch}] | step:{step:5d} | loss:{loss:6.4f} | "
            f"tokens:{tokens_trained_on*1e-9:6.4f}B | step time:{duration_ms:4.2f}ms ")

    if save_prefix is not None:
      eqx.tree_serialise_leaves(f"{save_prefix}.{total_step:06d}.model.eqx", model)
      eqx.tree_serialise_leaves(f"{save_prefix}.{total_step:06d}.opt_state.eqx", opt_state)

    print(f"Epoch[{epoch}] | step:{total_step} | loss acc:{loss_acc}")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--load_prefix", default=None, type=str)
  parser.add_argument("--save_prefix", default=None, type=str)
  args = parser.parse_args()

  train(num_epochs, args.load_prefix, args.save_prefix)
