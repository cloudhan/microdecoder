import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx
from data.sequence_reverse import get_model_context_len, bog_id, eog_id
from model.gpt2 import GPT2, GPT2Config

model = GPT2(
    GPT2Config(
        n_ctx=get_model_context_len(max_seq_len=64),
        n_vocab=12,
        n_layer=8,
        n_head=8,
        n_embd=32,
        dropout=0.0,
        vocab_round_up=1,  # dont roundup since sequence reverse has very little vocab
        inference=True,
    ),
    key=jax.random.PRNGKey(0),
)

load_prefix = "checkpoints/seqrev.006400"
model = eqx.tree_deserialise_leaves(f"{load_prefix}.model.eqx", model)
model = eqx.nn.inference_mode(model, True)


def infer_dyn_inc(model, input_ids_list):
  input_ids = jnp.array(input_ids_list, dtype=np.int32)
  new_token_id = jnp.array([bog_id])
  while input_ids.shape[-1] < 64 and new_token_id != eog_id:
    input_ids = jnp.concat([input_ids, new_token_id], axis=-1)
    with jax.default_matmul_precision("float32"):
      logits = model(input_ids, key=jax.random.PRNGKey(0))
    new_token_id = jnp.argmax(logits[-1], axis=-1, keepdims=True)
  return jnp.concat([input_ids, new_token_id], axis=-1), len(input_ids)


def infer_dyn_static_buffer(model, input_ids_list):
  input_ids = np.zeros(get_model_context_len(64), dtype=np.int32)
  input_ids[:len(input_ids_list) + 1] = np.concatenate([input_ids_list, np.array([bog_id])])
  input_ids = jnp.array(input_ids)
  idx = len(input_ids_list)
  while idx < 64 and input_ids[idx] != eog_id:
    with jax.default_matmul_precision("float32"):
      logits = model(input_ids, key=jax.random.PRNGKey(0))
    new_ids_at_idx = jnp.argmax(logits[idx], axis=-1)
    idx += 1
    input_ids = input_ids.at[idx].set(new_ids_at_idx)
  return input_ids, idx

def infer_jit(model, input_ids_list):

  def cond(xs):
    ids, idx = xs
    return jnp.logical_and(idx < 64, ids[idx] != eog_id)

  def body(xs):
    ids, idx = xs
    logits = model(ids, key=jax.random.PRNGKey(0))
    new_ids_at_idx = jnp.argmax(logits[idx], axis=-1)
    idx += 1
    ids = ids.at[idx].set(new_ids_at_idx)
    return ids, idx

  @eqx.filter_jit
  def do_infer(input_ids, idx):
    with jax.default_matmul_precision("float32"):
      output_ids, idx = jax.lax.while_loop(cond, body, (input_ids, idx))
    return output_ids, idx

  input_ids = np.ones(get_model_context_len(64), dtype=np.int32)
  input_ids[:len(input_ids_list) + 1] = np.concatenate([input_ids_list, np.array([bog_id])])
  output_ids, idx = do_infer(input_ids, len(input_ids_list))
  return output_ids, idx


input_ids_list = [2, 3, 4, 5]

infer = infer_dyn_static_buffer
output_ids, idx = infer(model, input_ids_list)
print("output buffer:", output_ids[:idx + 1])
print("input: ", input_ids_list)
print("output:", output_ids[len(input_ids_list) + 1:idx])
