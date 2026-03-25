import pandas as pd
import re

# Load existing CSV
df = pd.read_csv("finance_reuters_articles.csv")

print("Before cleaning:", len(df))

# Remove null / empty
df = df.dropna(subset=["text"])
df = df[df["text"].str.strip() != ""]

# Remove very short articles
df = df[df["text"].str.split().apply(len) > 30]

# Normalize text
def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.replace("Reuters", "")
    return text.strip()

df["text"] = df["text"].apply(clean_text)

# Clean title
df["title"] = df["title"].apply(lambda x: clean_text(x) if isinstance(x, str) else "")

# Remove duplicates
df = df.drop_duplicates(subset=["text"])

# Combine title + text 
df["document"] = "Title: " + df["title"] + ". " + df["text"]

print("After cleaning:", len(df))

# Save clean data
df.to_csv("finance_reuters_cleaned.csv", index=False)

print("Saved cleaned dataset!")