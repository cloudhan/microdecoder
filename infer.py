import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp


def infer(model, input_ids_list, context_len=80):
  length = len(input_ids_list)
  input_ids = np.zeros(context_len, dtype=np.int32)
  input_ids[:length] = np.array(input_ids_list)
  input_ids = jnp.array(input_ids)

  model = eqx.filter_jit(model)
  while length < context_len:
    with jax.default_matmul_precision("float32"):
      logits = model(input_ids, key=jax.random.PRNGKey(0))
    new_ids = jnp.argmax(logits[length - 1], axis=-1)
    input_ids = input_ids.at[length].set(new_ids)
    length += 1
  return input_ids, length


def infer_topk(model, input_ids_list, context_len=80, topk=5):

  def sample_step(input_ids, logits, logp, length):  # operate on a single seq
    new_logits = logits[length - 1]
    new_logp = jax.nn.log_softmax(new_logits)
    new_indices = jnp.argpartition(-new_logp, topk)[:topk]
    new_logp_topk = new_logp[new_indices]
    input_ids_topk = jnp.broadcast_to(input_ids, shape=(topk,) + input_ids.shape)
    input_ids = input_ids_topk.at[:, length].set(new_indices)
    logp += new_logp_topk
    return input_ids, logp

  length = len(input_ids_list)
  input_ids = np.zeros(context_len, dtype=np.int32)
  input_ids[:length] = np.array(input_ids_list)
  input_ids = jnp.array([input_ids])
  logp = jnp.zeros((1,), dtype=jnp.float32)

  # model = eqx.filter_jit(jax.vmap(model))
  model = jax.vmap(model)
  # sample_step = jax.jit(jax.vmap(sample_step, in_axes=(0,0,0,None)))
  sample_step = jax.vmap(sample_step, in_axes=(0, 0, 0, None))

  key = jax.random.PRNGKey(0)

  while length < context_len:
    with jax.default_matmul_precision("float32"):
      logits = model(input_ids, key=jax.random.split(key, logp.shape[0]))

    input_ids, logp = sample_step(input_ids, logits, logp, length)
    input_ids = jnp.reshape(input_ids, (-1, context_len))
    logp = jnp.reshape(logp, (-1,))

    indices = jnp.argsort(logp, descending=True)[:topk]
    input_ids = input_ids[indices]
    logp = logp[indices]

    length += 1
  return input_ids, length


if __name__ == "__main__":
  import tiktoken
  from model.gpt2_mixed import GPT2, GPT2_S

  import jax_utils

  model = GPT2(GPT2_S, key=jax.random.PRNGKey(0))
  model = jax_utils.cast_fp32(model, jnp.bfloat16)

  load_prefix = "checkpoints/gpt2.006400"
  model = eqx.tree_deserialise_leaves(f"{load_prefix}.model.eqx", model)
  model = eqx.nn.inference_mode(model, True)

  enc = tiktoken.get_encoding("gpt2")
  input_ids_list = enc.encode_ordinary("""First Citizen:\n""")

  context_len = 32

  jax.numpy.set_printoptions(linewidth=200)

  output_ids, length = infer(model, input_ids_list, context_len=context_len)
  print("output buffer:")
  print(output_ids[:length])
  print("input ids :", input_ids_list)
  print("output ids:", output_ids[len(input_ids_list):length])
  print("input :", repr(enc.decode(input_ids_list)))
  print("output:", repr(enc.decode(output_ids[len(input_ids_list):length])))
  print("=" * 80)

  topk = 8
  output_ids, length = infer_topk(model, input_ids_list, context_len=context_len, topk=topk)
  print("output buffer:")
  print(output_ids[:length])

  print("input ids:", input_ids_list)
  for i in range(topk):
    print(f"output[{i}] ids:", output_ids[i, len(input_ids_list):length])

  print("input:", repr(enc.decode(input_ids_list)))
  for i in range(topk):
    print(f"output[{i}]:", repr(enc.decode(output_ids[i, len(input_ids_list):length])))
