import jax
from jax import jit, vmap
import jax.numpy as jnp
from model.gpt2 import GPT2, GPT2Config
from data import sequence_reverse
from tqdm import tqdm
import optax
import equinox as eqx


def get_lr_schedule(base_lr, warmup_steps, train_steps):
  return optax.warmup_cosine_decay_schedule(
      init_value=1e-5 * base_lr,
      peak_value=base_lr,
      warmup_steps=warmup_steps,
      decay_steps=train_steps - warmup_steps,
  )


@eqx.filter_value_and_grad(allow_int=True)
def compute_loss(model, batch, key=None):
  logits = jax.vmap(model)(batch[:, :-1], key=jax.random.split(key, batch.shape[0]))
  labels = batch[:, 1:]
  # loss_fn is a mapping: ((batch_size, seq_len, vacab_size), (batch_size, seq_len)) -> (batch_size, seq_len)
  losses = optax.losses.softmax_cross_entropy_with_integer_labels(logits, labels)
  return losses.mean()


@eqx.filter_jit
def train_step(model, batch, optimizer, opt_state, key=None):
  loss, grad = compute_loss(model, batch, key=key)
  updates, opt_state = optimizer.update(grad, opt_state, model)
  # model = optax.apply_updates(model, updates)
  model = eqx.apply_updates(model, updates)
  return loss, model, opt_state


def train(num_epochs, load_prefix=None, save_prefix=None):
  key = jax.random.PRNGKey(42)
  model = GPT2(
      GPT2Config(
          n_ctx=sequence_reverse.get_model_context_len(max_seq_len=args.batch_size),
          n_vocab=12,
          n_layer=4,
          n_head=4,
          n_embd=64,
          dropout=0.05,
          vocab_round_up=1,  # dont roundup since sequence reverse has very little vocab
      ),
      key=key,
  )

  train_dataloader = sequence_reverse.get_dataloader(split="train", batch_size=args.batch_size)
  val_dataloader = sequence_reverse.get_dataloader(split="test_val", batch_size=args.batch_size)

  # optimizer = optax.adam(learning_rate=get_lr_schedule(0.01, 500, 50000))
  optimizer = optax.adam(learning_rate=0.003)
  opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

  if load_prefix is not None:
    model = eqx.tree_deserialise_leaves(f"{load_prefix}.model.eqx", model)
    opt_state = eqx.tree_deserialise_leaves(f"{load_prefix}.opt_state.eqx", opt_state)

  total_step = 0
  for epoch in tqdm(range(1, num_epochs + 1)):
    loss_acc = 0
    for step, batch in enumerate(train_dataloader):
      key, train_step_key = jax.random.split(key)
      loss, model, opt_state = train_step(model, batch, optimizer, opt_state, key=train_step_key)
      loss_acc += loss
      total_step += 1

    if save_prefix is not None:
      eqx.tree_serialise_leaves(f"{save_prefix}.{total_step:06d}.model.eqx", model)
      eqx.tree_serialise_leaves(f"{save_prefix}.{total_step:06d}.opt_state.eqx", opt_state)

    print(f"Epoch[{epoch}] step:{total_step} | accumulated train loss:{loss_acc}")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--load_prefix", default=None, type=str)
  parser.add_argument("--save_prefix", default=None, type=str)
  parser.add_argument("--batch_size", default=256, type=int)
  parser.add_argument("--num_epochs", default=25, type=int)
  args = parser.parse_args()

  train(args.num_epochs, args.load_prefix, args.save_prefix)
