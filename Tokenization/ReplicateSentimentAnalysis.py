import torch.nn.functional
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

rawInput = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"



def ClassifyByPipeline(Input, model = checkpoint) -> dict:
    classifier = pipeline("sentiment-analysis", model = model)
    return classifier(Input)

def Tokenize(Input, Checkpoint = checkpoint) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(Checkpoint)
    return tokenizer(Input, padding=True, truncation=True, return_tensors="pt")

# print(ClassifyByPipeline(rawInput))
inputs = Tokenize(rawInput)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim = -1)
print(predictions)
print(model.config.id2label)