import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def infer(model, input_ids_list, context_len=80):
  length = len(input_ids_list)
  input_ids = np.zeros(context_len, dtype=np.int32)
  input_ids[:length] = np.array(input_ids_list)
  input_ids = jnp.array(input_ids)

  while length < context_len:
    with jax.default_matmul_precision("float32"):
      logits = model(input_ids, key=jax.random.PRNGKey(0))
    new_ids = jnp.argmax(logits[length - 1], axis=-1)
    input_ids = input_ids.at[length].set(new_ids)
    length += 1
  return input_ids, length


@functools.partial(jax.vmap, in_axes=(0, 0, 0, None, None))
def sample_step(input_ids, logits, logp, length, topk):  # operate on a single seq
  new_logits = logits[length - 1]
  new_logp = jax.nn.log_softmax(new_logits)
  new_indices = jnp.argpartition(-new_logp, topk)[:topk]
  new_logp_topk = new_logp[new_indices]
  input_ids_topk = jnp.broadcast_to(input_ids, shape=(topk,) + input_ids.shape)
  input_ids = input_ids_topk.at[:, length].set(new_indices)
  logp += new_logp_topk
  return input_ids, logp


@functools.partial(jax.jit, static_argnums=(0, 4, 5))
def infer_topk_impl(static, params, input_ids, length, context_len, topk):
  model = eqx.combine(static, params)
  infer_keys = jax.random.split(jax.random.PRNGKey(0), topk)

  def cond(xs):
    input_ids, logp, length = xs
    return length < context_len

  def body(xs):
    input_ids, logp, length = xs
    with jax.default_matmul_precision("float32"):
      logits = model(input_ids, infer_keys)

    input_ids, logp = sample_step(input_ids, logits, logp, length, topk)
    input_ids = jnp.reshape(input_ids, (-1, context_len))
    logp = jnp.reshape(logp, (-1,))

    indices = jnp.argsort(logp, descending=True)[:topk]
    input_ids = input_ids[indices]
    logp = logp[indices]

    length += 1
    return input_ids, logp, length

  init_logp = jnp.array([0] + [-float("inf")] * (topk - 1), dtype=jnp.float32)
  return jax.lax.while_loop(cond, body, (input_ids, init_logp, length))


def infer_topk(model, input_ids_list, context_len=80, topk=2):
  length = len(input_ids_list)
  input_ids = np.zeros((topk, context_len), dtype=np.int32)
  input_ids[0, :length] = np.array(input_ids_list)
  input_ids = jnp.array(input_ids)

  params, static = eqx.partition(model, eqx.is_array)
  input_ids, logp, length = infer_topk_impl(static, params, input_ids, length, context_len, topk)
  return input_ids, logp, length


if __name__ == "__main__":
  import argparse
  import tiktoken
  from model.gpt2 import GPT2, GPT2_S

  import jax_utils

  jax_utils.config()

  parser = argparse.ArgumentParser()
  parser.add_argument("load_prefix")
  parser.add_argument("prompt")
  parser.add_argument("--context_len", type=int, default=32)
  parser.add_argument("--topk", type=int, default=2)
  args = parser.parse_args()

  GPT2_S.act_dtype = jnp.bfloat16
  GPT2_S.emb_dtype = jnp.float16

  model = GPT2(GPT2_S, key=jax.random.PRNGKey(0))
  model = jax_utils.cast_fp32(model, jnp.bfloat16)

  model = eqx.tree_deserialise_leaves(f"{args.load_prefix}.model.eqx", model)
  model = eqx.nn.inference_mode(model, True)

  enc = tiktoken.get_encoding("gpt2")
  input_ids_list = enc.encode_ordinary(args.prompt)

  jax.numpy.set_printoptions(linewidth=200)

  jit_model = eqx.filter_jit(model)
  output_ids, length = infer(jit_model, input_ids_list, context_len=args.context_len)
  output_ids = np.array(output_ids)

  print("output buffer:")
  print(output_ids[:length])
  print("input ids :", input_ids_list)
  print("output ids:", output_ids[len(input_ids_list):length])
  print("input :", repr(enc.decode(input_ids_list)))
  print("output:", repr(enc.decode(output_ids[len(input_ids_list):length])))
  print("=" * 80)

  jit_model = eqx.filter_jit(jax.vmap(lambda x, ks: model(x, key=ks)))
  output_ids, logp, length = infer_topk(jit_model, input_ids_list, context_len=args.context_len, topk=args.topk)
  output_ids = np.array(output_ids)
  logp = logp.tolist()

  print("output buffer:")
  print(output_ids[:length])

  print("input ids:", input_ids_list)
  for i in range(args.topk):
    print(f"output[{i}] logp:{logp[i]:6.2f} ids:", output_ids[i, len(input_ids_list):length])

  print("input:", repr(enc.decode(input_ids_list)))
  for i in range(args.topk):
    print(f"output[{i}]:", repr(enc.decode(output_ids[i, len(input_ids_list):length])))
