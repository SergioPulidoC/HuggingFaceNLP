import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
print(torch.softmax(output.logits, dim=-1))

# tokens = tokenizer.tokenize(sequence)
# ids = tokenizer.convert_tokens_to_ids(tokens)
#
# input_ids = torch.tensor([ids])
# # print("Input IDs:", input_ids)
#
# output = model(input_ids)
# # print("Logits:", output.logits)
#
# batchedIds = [ids, ids, ids]
#
# print(model(torch.tensor(batchedIds)))