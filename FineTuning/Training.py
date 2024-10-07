import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, \
    Trainer
from pathlib import Path

dataset_name = "mrpc"

raw_datasets = load_dataset("glue", dataset_name)
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

def get_sentence_fields(dataset):
    return [field for field in dataset.keys() if "sentence" in field]

def tokenize_dataset(dataset):
    sequences_names = get_sentence_fields(dataset)
    sequences = [dataset[field] for field in sequences_names]
    return tokenizer(*sequences, truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_dataset, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments("test-trainer")

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
trainer.save_model(Path() / "MrpcFinetune")
