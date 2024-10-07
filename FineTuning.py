from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

rawDataset = load_dataset("glue", "mrpc")
train = rawDataset["train"]
validation = rawDataset["validation"]
test = rawDataset["test"]

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def TokenizeDataset(dataset):
    return tokenizer(dataset["sentence1"], dataset["sentence2"], truncation=True)

tokenizedDatasets = rawDataset.map(TokenizeDataset, batched=True)

dataCollator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenizedDatasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
batch = dataCollator(samples)
print({k: v.shape for k, v in batch.items()})