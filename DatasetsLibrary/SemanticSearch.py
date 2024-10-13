import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel

issues_dataset = load_dataset("lewtun/github-issues", split="train")
issues_dataset = issues_dataset.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
)
columns = issues_dataset.column_names

##Remove irrelevant columns
relevant_columns = ["title", "body", "html_url", "comments"]
irrelevant_columns = set(relevant_columns).symmetric_difference(columns)
issues_dataset = issues_dataset.remove_columns(irrelevant_columns)

##Make one row for each comment (there are multiple comments in each issue)
print(issues_dataset["comments"][0])
issues_dataset.set_format("pandas")
df = issues_dataset[:]
comments_df = df.explode("comments", ignore_index=True)
# print(comments_df.head(2)["comments"])
comments_dataset = Dataset.from_pandas(comments_df)

##Remove rows with short comments
comments_dataset = comments_dataset.map(
    lambda x: {"comment_length": len(x["comments"].split())}
)
comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)

##Merge title, description and comments in a single column
def concatenate_text(examples):
    return {
        "text": examples["title"]
        + " \n "
        + examples["body"]
        + " \n "
        + examples["comments"]
    }
comments_dataset = comments_dataset.map(concatenate_text)

##Creating embeddings

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

device = torch.device("cuda")
model.to(device)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)

##Using FAISS

embeddings_dataset.add_faiss_index(column="embeddings")

question = "How can I load a dataset offline?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()

scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)

samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)

for _, row in samples_df.iterrows():
    print(f"COMMENT: {row.comments}")
    print(f"SCORE: {row.scores}")
    print(f"TITLE: {row.title}")
    print(f"URL: {row.html_url}")
    print("=" * 50)
    print()