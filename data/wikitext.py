from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
# from transformers import DataCollatorWithPadding
from transformers import DataCollatorForLanguageModeling
import jax_dataloader as jdl

dataset_name = "wikitext-2-v1"
# dataset_name = "wikitext-103-v1"

# Shoudld I use it in real training process? Will it affect the data feeding performance?
# streaming = True
streaming = False

batch_size = 32
max_context_length = 128

def get_dataset():
  dataset = load_dataset("Salesforce/wikitext", name=dataset_name, streaming=streaming)
  merged_test_val = concatenate_datasets([dataset["test"], dataset["validation"]])
  dataset.pop("test")
  dataset.pop("validation")
  dataset["val"] = merged_test_val
  return dataset


dataset = get_dataset()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
  tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})


def process(example):
  return tokenizer(
      example["text"],
      padding="max_length",
      truncation=True,
      max_length=max_context_length,
      return_attention_mask=True,
  )


tokenized_datasets = dataset.map(process, batched=True).with_format("jax")
# collator = DataCollatorWithPadding(tokenizer=tokenizer)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_dataloader = jdl.DataLoader(
    tokenized_datasets["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collator,
    backend="jax",
)

print(train_dataloader)
for idx, batch in enumerate(train_dataloader):
  if idx > 2:
    break
  print(batch.keys())
  print(batch["input_ids"].shape, batch["attention_mask"].shape)
