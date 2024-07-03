from dataclasses import dataclass

import jax
import jax.experimental
import jax.numpy as jnp
from jax import vmap
from jaxtyping import PRNGKeyArray
import equinox as eqx
import equinox.nn as nn

# [1] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language Models are
#     Unsupervised Multitask Learners. Retrieved from https://openai.com/research/better-language-models
#
# [2] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia
#     Polosukhin. 2017. Attention is All you Need. In Proceedings of the 31st International Conference on Neural
#     Information Processing Systems (NIPS’17), 2017. Curran Associates Inc., Long Beach, California, USA, 6000–6010.
#     Retrieved from
#     https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
#
# [3] https://www.lesswrong.com/posts/pvCpvMjjyXaAXtnSi/because-of-layernorm-directions-in-gpt-2-mlp-layers-are
#
# [4] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. Improving Language Understanding by
# Generative Pre-Training. Retrieved from https://openai.com/research/language-unsupervised
#
# [5] https://www.reddit.com/r/MachineLearning/comments/iifw9h/r_gpt2_position_embeddings_visualized/

# About Dropout
#
# By [1], sub-layers are multi-head self-attention and positionwise fully connected feed-forward networt, that is ,
# {MHA, FFN}. "We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and
# normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the
# encoder and decoder stacks."

# About LayerNorm
#
# By [2], "... with a few modifications. Layer normalization was moved to the input of each sub-block ... and an
# additional layer normalization was added after the final self-attention block."

# About Positional Encoding
#
# By [4] (aka, GPT1 paper), "We used learned position embeddings instead of the sinusoidal version proposed in the
# original work."
#
# By [5] "... Also curious is that I'm able to train a smaller GPT-2 without position embeddings with no ill effects on
# test loss. 8 layers, 8 heads, 512 model depth, 512 context, 100 million tokens of WebText2. Both with and without
# position embeddings I get a test loss of ~4.9. That said, I'm not sure what effect that ablation would have on larger
# models or longer training;"


@dataclass
class GPT2Config:
  n_ctx: int
  n_vocab: int
  n_layer: int
  n_head: int
  n_embd: int
  dropout: float
  inference: bool = False


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
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, key=key)
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
  dropout: nn.Dropout

  def __init__(self, config: GPT2Config, *, key: PRNGKeyArray | None = None):
    self.dropout = nn.Dropout(config.dropout, inference=config.inference)

  def __call__(
      self,
      q,  # [seqlen_q, num_head, head_dim]
      k,  # [seqlen_kv, num_head, head_dim]
      v,  # [seqlen_kv, num_head, head_dim]
      *,
      key: PRNGKeyArray | None = None):
    s, h, d = q.shape
    t, h, d = k.shape  # and v.shape

    scale = 1.0 / jnp.sqrt(d)
    x = jnp.einsum("shd,thd -> hst", q, k).astype(jnp.float32)
    x = jax.nn.softmax(x * scale).astype(q.dtype)
    x = jnp.einsum("hst,thd -> shd", x, v).reshape(s, h * d)
    # NOTE: in paper, W^O, the output projection is not omitted. but is fused into V.
    # Both V and W^O are Linear, thus can be written as a single Linear, mathematically.
    x = self.dropout(x)
    return x


# https://github.com/openai/gpt-2/blob/9b63575ef4/src/model.py#L115-L120
class FFN(eqx.Module):
  c_fc: nn.Linear
  c_proj: nn.Linear
  dropout: nn.Dropout

  def __init__(self, config: GPT2Config, *, key: PRNGKeyArray | None = None):
    fc_key, proj_key = jax.random.split(key)
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, key=fc_key)
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, key=proj_key)
    self.dropout = nn.Dropout(config.dropout, inference=config.inference)

  def __call__(self, x, *, key: PRNGKeyArray | None = None):

    def call_on_single_x(x, *, key: PRNGKeyArray | None = None):
      x = self.c_fc(x)
      x = jax.nn.gelu(x)
      x = self.c_proj(x)
      x = self.dropout(x)
      return x

    return vmap(call_on_single_x)(x)


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
    self.ln1 = nn.LayerNorm(config.n_embd)
    self.ln2 = nn.LayerNorm(config.n_embd)
    self.ffn = FFN(config, key=ffn_key)

  def __call__(self, x, *, key: PRNGKeyArray | None = None):
    # NOTE: the layer norm is moved
    attn_key, ffn_key = jax.random.split(key)
    x = vmap(self.ln1)(x)
    q, k, v = self.proj(x)
    x = x + self.attn(q, k, v, key=attn_key)
    x = vmap(self.ln2)(x)
    x = x + self.ffn(x, key=ffn_key)
    return x


class GPT2(eqx.Module):
  wpe: nn.Embedding  # position embedding
  wte: nn.Embedding  # word (token) embedding
  dropout: nn.Dropout
  decoder_blocks: nn.Sequential
  ln_f: nn.LayerNorm
  lm_head: nn.Linear

  def __init__(self, config: GPT2Config, *, key: PRNGKeyArray | None = None):

    def ceil_div(x, y):
      return ((x - 1) // y + 1) * y

    wpe_key, wte_key, head_key, *keys = jax.random.split(key, num=3 + config.n_layer)
    self.wpe = nn.Embedding(config.n_ctx, config.n_embd, key=wpe_key)
    self.wte = nn.Embedding(ceil_div(config.n_vocab, 8), config.n_embd, key=wte_key)
    self.dropout = nn.Dropout(config.dropout, inference=config.inference)
    self.decoder_blocks = nn.Sequential([DecoderBlock(config, key=keys[i]) for i in range(config.n_layer)])
    self.ln_f = nn.LayerNorm(config.n_embd)
    self.lm_head = nn.Linear(config.n_embd, ceil_div(config.n_vocab, 8), key=head_key)

  def __call__(self, input_ids, position_ids=None, attention_mask=None, *, key: PRNGKeyArray | None = None):
    key, dropout_key = jax.random.split(key)

    if position_ids is None:
      position_ids = self._gen_position_ids(input_ids.shape[-1])

    if attention_mask is None:
      attention_mask = self._gen_attention_mask(input_ids.shape[-1])

    tok_emb = vmap(self.wpe)(input_ids)
    pos_emb = vmap(self.wpe)(position_ids)
    x = self.dropout(tok_emb + pos_emb, key=dropout_key)
    x = self.decoder_blocks(x, key=key)
    x = vmap(self.ln_f)(x)
    logits = vmap(self.lm_head)(x)
    return logits

  def _gen_position_ids(self, length):
    position_id = jnp.arange(length)
    return position_id

  def _gen_attention_mask(self, length):
    # raise NotImplementedError
    return None


if __name__ == "__main__":
  config = GPT2Config(n_ctx=1024, n_vocab=50304, n_layer=12, n_head=12, n_embd=768, dropout=0.0)
  key = jax.random.PRNGKey(0)

  gpt2 = GPT2(config, key=key)
  input_ids = jnp.repeat(jnp.array([[0, 1, 2, 3, 5]]), 4, axis=0)

  batch_gpt2 = vmap(gpt2)
  batch_gpt2 = jax.jit(batch_gpt2)
  import time
  start = time.time()
  for i in range(128):
    print(i)
    output = batch_gpt2(input_ids, key=jax.random.split(key, input_ids.shape[0]))
    print(output.shape)
  end = time.time()
  print(end - start)
