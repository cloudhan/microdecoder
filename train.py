import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import time
import tiktoken

import jax_utils

from model.gpt2 import GPT2, GPT2_S
from infer import infer_topk

batch_size = 480
mini_batch_size = 12
context_len = 1024

mini_steps = batch_size // mini_batch_size
assert batch_size % mini_batch_size == 0

decode_interval = 200
loss_acc_interval = 350
checkpoint_interval = 500


# dataset = "tinyshakespeare"
dataset = "finewebedu"

if dataset == "tinyshakespeare":
  from data.tinyshakespeare import get_dataloader
  visualize_prompt = "First Citizen:\n"
if dataset == "finewebedu":
  from data.finewebedu import get_dataloader
  # visualize_prompt = "JAX is a Python library "
  visualize_prompt = "A language model is a "


def infer_print(model):
  enc = tiktoken.get_encoding("gpt2")
  input_ids_list = enc.encode_ordinary(visualize_prompt)

  topk = 4
  output_ids, length = infer_topk(model, input_ids_list, context_len=64, topk=topk)
  print("input:", repr(enc.decode(input_ids_list)))
  for i in range(topk):
    print(f"output[{i}]:", repr(enc.decode(output_ids[i, len(input_ids_list):length])))


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
  losses = optax.losses.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), label_ids)
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


def to_mini_batches(batch, mini_batch_size, mini_steps):
  x, y = batch
  return (
      x.reshape(mini_steps, mini_batch_size, x.shape[-1]),
      y.reshape(mini_steps, mini_batch_size, y.shape[-1]),
  )


def train(load_prefix=None, save_prefix=None, mixed_precision=True):
  key = jax.random.PRNGKey(42)

  if mixed_precision:
    GPT2_S.act_dtype = jnp.bfloat16
    GPT2_S.emb_dtype = jnp.float16
  model = GPT2(GPT2_S, key=key)

  if mixed_precision:
    model = jax_utils.cast_fp32(model, jnp.bfloat16)

  jax_utils.count_params(model)
  jax_utils.count_decay_non_decay_params(model, jax_utils.is_decayable)

  optimizer = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(
          learning_rate=get_lr_schedule(6e-4, 6e-5, 2000, 19000),
          b1=0.9,
          b2=0.95,
          eps=1e-5,
          weight_decay=0.1,
          mask=jax_utils.is_decayable,
          # NOTE: in mixed precision, acc dtype requires dynamic range to be sufficient high.
          # default to None cause wte and wpe's acc dtype to be fp16 and then preduce NaN after iteration.
          mu_dtype=jnp.bfloat16 if mixed_precision else None,
      ),
  )
  opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

  if load_prefix is not None:
    model = eqx.tree_deserialise_leaves(f"{load_prefix}.model.eqx", model)
    opt_state = eqx.tree_deserialise_leaves(f"{load_prefix}.opt_state.eqx", opt_state)
  model = eqx.nn.inference_mode(model, False)

  num_tokens_per_batch = batch_size * context_len
  tokens_trained_on = 0
  loss_acc = 0
  step = 0

  step_start_time, step_end_time = time.time(), time.time()
  for step, batch in enumerate(get_dataloader(batch_size, context_len)):
    if step % decode_interval == 0:
      infer_print(model)

    if step % loss_acc_interval == 0:
      print(f"step:{step:5d} | loss acc ({loss_acc_interval} steps):{loss_acc}")
      loss_acc = 0

    if step % checkpoint_interval == 0 and step != 0 and save_prefix is not None:
        eqx.tree_serialise_leaves(f"{save_prefix}.{step:06d}.model.eqx", model)
        eqx.tree_serialise_leaves(f"{save_prefix}.{step:06d}.opt_state.eqx", opt_state)

    mini_batches = to_mini_batches(batch, mini_batch_size, mini_steps)
    key, train_step_key = jax.random.split(key)
    loss, model, opt_state = train_step(model, mini_batches, optimizer, opt_state, key=train_step_key)
    duration_ms = (step_end_time - step_start_time) * 1000
    print(f"step:{step:5d} | loss:{loss:6.4f} | tokens:{tokens_trained_on*1e-9:6.4f}B | step time:{duration_ms:4.2f}ms ")

    # ------------------------------------
    tokens_trained_on += num_tokens_per_batch
    step_start_time, step_end_time = step_end_time, time.time()
    step += 1
    loss_acc += loss



if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--load_prefix", default=None, type=str)
  parser.add_argument("--save_prefix", default=None, type=str)
  parser.add_argument("--no_mixed_precision", default=False, action="store_true")
  args = parser.parse_args()

  train(args.load_prefix, args.save_prefix, mixed_precision=not args.no_mixed_precision)
