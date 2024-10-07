from pathlib import Path

from transformers import BertConfig, BertModel, AutoTokenizer, BertTokenizer

checkpoint = "bert-base-cased"

newBertModel = BertModel.from_pretrained(checkpoint)
#Saving a model
newBertModel.save_pretrained(Path() / "BertModel")
tokenizer = BertTokenizer.from_pretrained(checkpoint)

#Saving a tokenizer
tokenizer.save_pretrained(Path() / "BertTokenizer")


## Tokenize:
#Whole process
sequence = "Using a Transformer network is simple"
res = tokenizer(sequence)
print(res)
Adam

#By parts
# tokens = tokenizer.tokenize(sequence)
# # print(tokens)
# ids = tokenizer.convert_tokens_to_ids(tokens)
# # print(ids)
#
# ## Decode:
# #Whole process
# decodeRes = tokenizer.decode(res["input_ids"])
# # print(decodeRes)
#
# #By parts
# tokensFromIds = tokenizer.convert_ids_to_tokens(ids)
# print(tokensFromIds)
# reconstructed = tokenizer.convert_tokens_to_string(tokensFromIds)
# print(reconstructed)