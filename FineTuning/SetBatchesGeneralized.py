from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

datasetName = "mrpc"

rawDataset = load_dataset("glue", datasetName)

def get_sentence_fields(dataset):
    return [field for field in dataset.keys() if "sentence" in field]

def tokenize_dataset(dataset):
    sequences_names = get_sentence_fields(dataset)
    sequences = [dataset[field] for field in sequences_names]
    return tokenizer(*sequences, truncation=True)

tokenizedDatasets = rawDataset.map(tokenize_dataset, batched=True)
dataCollator = DataCollatorWithPadding(tokenizer=tokenizer)

train = tokenizedDatasets["train"]
print(train)
samples = train[:8]
sentence_names = get_sentence_fields(samples)
print(sentence_names)
samples = {k: v for k, v in samples.items() if k not in ["idx", *sentence_names]}
batch = dataCollator(samples)
# print({k: v.shape for k, v in batch.items()})