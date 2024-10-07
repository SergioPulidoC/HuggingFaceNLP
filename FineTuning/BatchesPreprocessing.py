from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

rawDataset = load_dataset("glue", "mrpc")
rawTrain = rawDataset["train"]
rawValidation = rawDataset["validation"]
rawTest = rawDataset["test"]
# print(train[15])
inputs = tokenizer(rawTrain["sentence1"], rawTrain["sentence2"])
# print("-----")
# print(inputs)
#
def TokenizeDataset(dataset):
    return tokenizer(dataset["sentence1"], dataset["sentence2"], truncation=True)

tokenizedDatasets = rawDataset.map(TokenizeDataset, batched=True)
dataCollator = DataCollatorWithPadding(tokenizer=tokenizer)

train = tokenizedDatasets["train"]
samples = train[:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
batch = dataCollator(samples)
print({k: v.shape for k, v in batch.items()})