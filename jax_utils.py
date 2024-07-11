import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from rich.console import Console
from rich.table import Table
import rich.text

def print_fwd_bwd(f, *args, jupyter=False, **kwargs) -> None:
  args, in_tree = jtu.tree_flatten((args, kwargs))

  def f_(*args):
    args, kwargs = jtu.tree_unflatten(in_tree, args)
    return f(*args, **kwargs)

  fwd = jax.make_jaxpr(lambda *args: jax.vjp(f_, *args))(*args).jaxpr

  y, f_vjp = jax.vjp(f_, *args)
  res, in_tree = jtu.tree_flatten(f_vjp)

  def g_(*args):
    *res, y = args
    f_vjp = jtu.tree_unflatten(in_tree, res)
    return f_vjp(y)

  bwd = jax.make_jaxpr(g_)(*res, y).jaxpr

  if jupyter:
    table = Table(show_header=False, show_lines=True, padding=(1, 2, 0, 2), box=None)
    table.add_row("[bold green]forward computation:",
                  "[bold green]backward computation:")
    table.add_row(rich.text.Text.from_ansi(str(fwd)),
                  rich.text.Text.from_ansi(str(bwd)))
    console = Console(width=240, force_jupyter=True)
    console.print(table)
  else:
    print("=" * 40, "fwd", "=" * 40)
    print(fwd)
    print("=" * 40, "bwd", "=" * 40)
    print(bwd)


def count_params(model) -> None:
  weights = eqx.filter(model, eqx.is_array)
  param_count = sum(x.size for x in jax.tree_util.tree_leaves(weights))
  print(f"parameters: {param_count * 1e-6:6.2f}M ({param_count})")


def count_decay_non_decay_params(model, is_decayable) -> None:
  decay, non_decay = eqx.partition(model, is_decayable(model))
  print(decay)
  print(non_decay)
  decay = jax.tree_util.tree_leaves(eqx.filter(decay, eqx.is_array))
  non_decay = jax.tree_util.tree_leaves(eqx.filter(non_decay, eqx.is_array))

  num_decay_t = len(decay)
  num_decay_p = sum(x.size for x in decay)

  num_non_decay_t = len(non_decay)
  num_non_decay_p = sum(x.size for x in non_decay)

  print(f"num decayed parameter tensors: {num_decay_t}, with {num_decay_p} parameters")
  print(f"num non-decayed parameter tensors: {num_non_decay_t}, with {num_non_decay_p} parameters")


def cast_emb(model, dtype):
  pass
  # print(jax.tree_util.tree_map(id, model))

  # key_leaves, tree_def = jax.tree_util.tree_flatten_with_path(model)
  # print(tree_def)
  # for idx, kl in enumerate(key_leaves):
  #   print(tree_def[idx], kl[0])


def cast_fp32(model, dtype):
  return jax.tree_util.tree_map(lambda x: x.astype(dtype) if eqx.is_array(x) and x.dtype == jnp.float32 else x, model)


is_decayable = functools.partial(jax.tree_util.tree_map, lambda x: eqx.is_array(x) and x.ndim >= 2)
