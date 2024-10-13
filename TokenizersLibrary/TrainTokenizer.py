from datasets import load_dataset
from transformers import AutoTokenizer

raw_datasets = load_dataset("code_search_net", "python", trust_remote_code=True)
def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )
training_corpus = get_training_corpus()

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

# oldTokens = old_tokenizer.tokenize(example)
# print(oldTokens)
#
# tokens = tokenizer.tokenize(example)
# print(tokens)
tokenizer.save_pretrained("code-search-net-tokenizer")