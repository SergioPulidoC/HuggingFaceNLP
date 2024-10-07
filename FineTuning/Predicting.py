import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, DataCollatorWithPadding

rawDataset = load_dataset("glue", "mrpc")

model = AutoModelForSequenceClassification.from_pretrained("MrpcFinetune")
tokenizer = AutoTokenizer.from_pretrained("MrpcFinetune")

def get_sentence_fields(dataset):
    return [field for field in dataset.keys() if "sentence" in field]

def tokenize_dataset(dataset):
    sequences_names = get_sentence_fields(dataset)
    sequences = [dataset[field] for field in sequences_names]
    return tokenizer(*sequences, truncation=True)

tokenized_datasets = rawDataset.map(tokenize_dataset, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,  # The trained model
    data_collator=data_collator,
    tokenizer=tokenizer,
)
predictions = trainer.predict(tokenized_datasets["validation"])
preds = np.argmax(predictions.predictions, axis=-1)
metric = evaluate.load("glue", "mrpc")
acc = metric.compute(predictions=preds, references=predictions.label_ids)
print(acc)