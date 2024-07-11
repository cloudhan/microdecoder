from dataclasses import dataclass

import functools
import math

import equinox as eqx
import equinox.nn as nn
import jax
import jax.experimental
import jax.numpy as jnp
from jax import vmap
from jaxtyping import PRNGKeyArray


@dataclass
class GPT2Config:
  n_ctx: int
  n_vocab: int
  n_layer: int
  n_head: int
  n_embd: int
  dropout: float
  bias: bool = False  # Linear and LayerNorm
  inference: bool = False
  vocab_round_up: int = 8


GPT2_S = GPT2Config(1024, 50257, 12, 12, 768, 0.0)
GPT2_M = GPT2Config(1024, 50257, 24, 16, 1024, 0.0)
GPT2_L = GPT2Config(1024, 50257, 36, 20, 1280, 0.0)
GPT2_XL = GPT2Config(1024, 50257, 48, 25, 1600, 0.0)


def init(pytree, key, **kwargs):
  for attr, init_method in kwargs.items():
    key, init_key = jax.random.split(key)
    pytree = eqx.tree_at(
        lambda t: getattr(t, attr, None),
        pytree,
        replace_fn=functools.partial(init_method, key=init_key),
    )
  return pytree


def jax_init_wrapper(jax_init, optional=False):
  """convert jax.nn initializer instance to accept array, instead of shape and dtype"""

  def eqx_init(w, key=None):
    if optional and w is None:
      return None
    return jax_init(key, w.shape, w.dtype)

  return eqx_init


normal_001 = jax_init_wrapper(jax.nn.initializers.normal(0.01))
normal_002 = jax_init_wrapper(jax.nn.initializers.normal(0.02))
normal_gpt = lambda n_layer: jax_init_wrapper(jax.nn.initializers.normal(0.02 / math.sqrt(2 * n_layer)))
zero = jax_init_wrapper(jax.nn.initializers.constant(0.0), optional=True)


class QKVProj(eqx.Module):
  c_attn: nn.Linear
  num_head: int
  head_dim: int

  def __init__(self, config: GPT2Config, *, key: PRNGKeyArray | None = None):
    key, _ = jax.random.split(key)
    # projection operate on head-wise. This should have been as
    #
    # self.c_attns = [nn.Linear(head_dim, 3 * head_dim, key=key[h]) for h in range(num_head)]
    # and then split 3*head_dim output to q, k and v.
    #
    # We cannot use vmap because weights are not share between heads, a.k.a., the functions are different.
    # We need *parallelly* apply multiple functions, not vmap with a single function.
    #
    # The following premature optimization fuse all thoes Linears into a single one.
    self.c_attn = init(nn.Linear(config.n_embd, 3 * config.n_embd, use_bias=config.bias, key=key), key, weight=normal_002, bias=zero)
    self.num_head = config.n_head
    self.head_dim = config.n_embd // config.n_head

  def __call__(self, x):

    def project_single_x(x):  # each x is H*D
      x = self.c_attn(x)
      # assuming dim of q, k and v are the same
      dim = x.shape[-1] // 3
      q = x[0 * dim:1 * dim].reshape(self.num_head, self.head_dim)
      k = x[1 * dim:2 * dim].reshape(self.num_head, self.head_dim)
      v = x[2 * dim:3 * dim].reshape(self.num_head, self.head_dim)
      return q, k, v

    return vmap(project_single_x)(x)


class MHA(eqx.Module):
  attn_dropout: nn.Dropout
  resid_dropout: nn.Dropout
  c_proj: nn.Linear  # output projection

  def __init__(self, config: GPT2Config, *, key: PRNGKeyArray | None = None):
    self.attn_dropout = nn.Dropout(config.dropout, inference=config.inference)
    self.resid_dropout = nn.Dropout(config.dropout, inference=config.inference)
    self.c_proj = init(nn.Linear(config.n_embd, config.n_embd, use_bias=config.bias, key=key), key, weight=normal_gpt(config.n_layer), bias=zero)

  def __call__(
      self,
      q,  # [seqlen_q, num_head, head_dim]
      k,  # [seqlen_kv, num_head, head_dim]
      v,  # [seqlen_kv, num_head, head_dim]
      *,
      key: PRNGKeyArray | None = None):
    attn_key, residual_key = jax.random.split(key, 2)
    s, h, d = q.shape
    t, h, d = k.shape  # and v.shape
    causal_bias = self.causal_mask(s, t) * -10000.0

    scale = 1.0 / jnp.sqrt(d)

    x = jnp.einsum("shd,thd -> hst", q, k).astype(jnp.float32)
    x = x * scale + causal_bias
    x = jax.nn.softmax(x).astype(q.dtype)
    x = self.attn_dropout(x, key=attn_key)
    x = jnp.einsum("hst,thd -> shd", x, v).reshape(s, h * d)
    # NOTE: in some paper, W^O, the output projection sometime is omitted. Because it is fused into V.
    # Both V and W^O are Linear, thus can be written as a single Linear, mathematically.
    x = self.resid_dropout(vmap(self.c_proj)(x), key=residual_key)

    return x

  def causal_mask(self, s, t):
    s = jnp.arange(s)[:, None]
    t = jnp.arange(t)[None, :]
    return (s < t)


# https://github.com/openai/gpt-2/blob/9b63575ef4/src/model.py#L115-L120
class FFN(eqx.Module):
  c_fc: nn.Linear
  c_proj: nn.Linear
  dropout: nn.Dropout

  def __init__(self, config: GPT2Config, *, key: PRNGKeyArray | None = None):
    fc_key, proj_key = jax.random.split(key)
    self.c_fc = init(nn.Linear(config.n_embd, 4 * config.n_embd, use_bias=config.bias, key=fc_key), key, weight=normal_002, bias=zero)
    self.c_proj = init(nn.Linear(4 * config.n_embd, config.n_embd, use_bias=config.bias, key=proj_key), key, weight=normal_gpt(config.n_layer), bias=zero)
    self.dropout = nn.Dropout(config.dropout, inference=config.inference)

  def __call__(self, x, *, key: PRNGKeyArray | None = None):

    def call_on_single_x(x, *, key: PRNGKeyArray | None = None):
      x = self.c_fc(x)
      x = jax.nn.gelu(x.astype(jnp.float32)).astype(jnp.bfloat16)
      x = self.c_proj(x)
      x = self.dropout(x, key=key)
      return x

    return vmap(call_on_single_x)(x, key=jax.random.split(key, x.shape[0]))


class DecoderBlock(eqx.Module):
  ln1: nn.LayerNorm
  proj: QKVProj
  attn: MHA
  ln2: nn.LayerNorm
  ffn: FFN

  def __init__(self, config: GPT2Config, *, key: PRNGKeyArray | None = None):
    proj_key, attn_key, ffn_key = jax.random.split(key, 3)
    self.proj = QKVProj(config, key=proj_key)
    self.attn = MHA(config, key=attn_key)
    self.ln1 = nn.LayerNorm(config.n_embd, use_bias=config.bias, dtype=jnp.bfloat16)
    self.ln2 = nn.LayerNorm(config.n_embd, use_bias=config.bias, dtype=jnp.bfloat16)
    self.ffn = FFN(config, key=ffn_key)

  def __call__(self, x, *, key: PRNGKeyArray | None = None):
    # NOTE: the layer norm is moved
    attn_key, ffn_key = jax.random.split(key)
    q, k, v = self.proj(vmap(self.ln1)(x.astype(jnp.float32)).astype(jnp.bfloat16))
    x = x + self.attn(q, k, v, key=attn_key)
    x = x + self.ffn(vmap(self.ln2)(x.astype(jnp.float32)).astype(jnp.bfloat16), key=ffn_key)
    return x


class GPT2(eqx.Module):
  wpe: nn.Embedding  # position embedding
  wte: nn.Embedding  # word (token) embedding
  dropout: nn.Dropout
  decoder_blocks: nn.Sequential
  ln_f: nn.LayerNorm
  # lm_head: nn.Linear
  # shared_wte_and_lm_head: nn.Shared

  def __init__(self, config: GPT2Config, *, key: PRNGKeyArray | None = None):

    def multiple_of(x, y):
      return ((x - 1) // y + 1) * y

    physical_num_vocabs = multiple_of(config.n_vocab, config.vocab_round_up)

    wpe_key, wte_key, head_key, *keys = jax.random.split(key, num=3 + config.n_layer)
    self.wpe = init(nn.Embedding(config.n_ctx, config.n_embd, dtype=jnp.float16, key=wpe_key), wpe_key, weight=normal_001)
    self.wte = init(nn.Embedding(physical_num_vocabs, config.n_embd, dtype=jnp.float16, key=wte_key), wte_key, weight=normal_002)
    self.dropout = nn.Dropout(config.dropout, inference=config.inference)
    self.decoder_blocks = nn.Sequential([DecoderBlock(config, key=keys[i]) for i in range(config.n_layer)])
    self.ln_f = nn.LayerNorm(config.n_embd, use_bias=config.bias, dtype=jnp.bfloat16)
    # self.lm_head = nn.Linear(config.n_embd, physical_num_vocabs, use_bias=False, key=head_key)


  def __call__(self, input_ids, position_ids=None, attention_mask=None, *, key: PRNGKeyArray | None = None):
    key, dropout_key = jax.random.split(key)

    if position_ids is None:
      position_ids = self._gen_position_ids(input_ids.shape[-1])

    if attention_mask is None:
      attention_mask = self._gen_attention_mask(input_ids.shape[-1])

    tok_emb = vmap(self.wte)(input_ids)
    pos_emb = vmap(self.wpe)(position_ids)
    x = self.dropout(tok_emb + pos_emb, key=dropout_key)
    x = self.decoder_blocks(x, key=key)
    x = vmap(self.ln_f)(x.astype(jnp.float32)).astype(jnp.bfloat16)
    logits = x @ self.wte.weight.T.astype(jnp.bfloat16)  # lm_head share wte weight
    return logits

  def _gen_position_ids(self, length):
    position_id = jnp.arange(length)
    return position_id

  def _gen_attention_mask(self, length):
    # raise NotImplementedError
    return None


if __name__ == "__main__":
  import optax
  import jax_utils

  config = GPT2Config(n_ctx=42, n_vocab=3000, n_layer=1, n_head=2, n_embd=128, dropout=0.0)
  key = jax.random.PRNGKey(0)

  batch_size = 4
  seq_len = 42

  print(f"batch_size:{batch_size}, seq_len:{seq_len}, num_vocab:{config.n_vocab}")

  model = GPT2(config, key=key)
  model = jax_utils.cast_fp32(model, jnp.bfloat16)

  @eqx.filter_jit
  @eqx.filter_value_and_grad(allow_int=True)
  def compute_loss(model, params, batch, key=None):
    model = eqx.combine(params, model)
    input_ids, label_ids = batch
    logits = jax.vmap(model)(input_ids, key=jax.random.split(key, input_ids.shape[0]))
    losses = optax.losses.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), label_ids)
    return losses.mean()

  key = jax.random.PRNGKey(0)
  input_ids = jnp.zeros((4, 42), dtype=jnp.int32)
  label_ids = jnp.ones((4, 42), dtype=jnp.int32)

  params, model = eqx.partition(model, eqx.is_array)
  f = functools.partial(compute_loss, model, key=key)
  jax_utils.print_fwd_bwd(f, params, (input_ids, label_ids))
