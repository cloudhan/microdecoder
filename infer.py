import jax
import jax.numpy as jnp
import equinox as eqx
from data.sequence_reverse import get_model_context_len, bog_id, eog_id
from model.gpt2 import GPT2, GPT2Config

model = GPT2(
    GPT2Config(
        n_ctx=get_model_context_len(max_seq_len=64),
        n_vocab=12,
        n_layer=4,
        n_head=1,
        n_embd=32,
        dropout=0.05,
        vocab_round_up=1,  # dont roundup since sequence reverse has very little vocab
        inference=True,
    ),
    key=jax.random.PRNGKey(0),
)

load_prefix = "checkpoints/seqrev.002560"
model = eqx.tree_deserialise_leaves(f"{load_prefix}.model.eqx", model)


# @eqx.filter_jit
def infer(model, input_ids):
  bog_token = jnp.array([bog_id])

  # def cond(xs):
  #   input_ids, new_token_id = xs
  #   return input_ids.shape[-1] < 64
  #   # return new_token_id != eog_id or input_ids.shape[-1] < 64

  # def body(xs):
  #   input_ids, new_token_id = xs
  #   input_ids = jnp.concat([input_ids, new_token_id], axis=-1)
  #   logits = model(input_ids, key=jax.random.PRNGKey(0))
  #   new_token_id = jnp.argmax(logits[-1], axis=-1, keepdims=True)
  #   return input_ids, new_token_id

  # output_ids, new_token_id = jax.lax.while_loop(cond, body, (input_ids, bog_token))
  # return jnp.concat([output_ids, new_token_id], axis=-1)

  new_token_id = bog_token
  while new_token_id != eog_id and input_ids.shape[-1] < 64:
    input_ids = jnp.concat([input_ids, new_token_id], axis=-1)
    print(input_ids)
    logits = model(input_ids, key=jax.random.PRNGKey(0))
    new_token_id = jnp.argmax(logits[-1], axis=-1, keepdims=True)
  return jnp.concat([input_ids, new_token_id], axis=-1)


infer(model, jnp.array([2,3,4,5]))
