import pprint

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# example = ("Today I'm going to the pool. "
#            "The pool is located in Compensar.")
# encoding = tokenizer(example)
# print(encoding.tokens())
# print(encoding.sequence_ids())


model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

example = "My name is Sergio, I live in Bogotá. I used to work in Hyzca Studios but it sucked."
inputs = tokenizer(example, return_tensors="pt", return_offsets_mapping=True)
outputs = model(**inputs)


probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
predictions = outputs.logits.argmax(dim=-1)[0].tolist()

results = []
tokens = inputs.tokens()

for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        results.append(
            {"entity": label, "score": probabilities[idx][pred], "word": tokens[idx]}
        )

pprint.pprint(results)