import os
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

DATA_DIR = "."

finance_topics = [
    "earn",
    "acq",
    "money-fx",
    "trade",
    "interest",
    "crude"
]

articles = []

for file in tqdm(os.listdir(DATA_DIR)):
    if file.endswith(".sgm"):

        with open(file, "r", encoding="latin-1") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

            for item in soup.find_all("reuters"):

                title = item.title.text if item.title else ""
                body = item.body.text if item.body else ""

                topics = []
                if item.topics:
                    topics = [d.text for d in item.topics.find_all("d")]

                if any(t in finance_topics for t in topics):

                    if body.strip():
                        articles.append({
                            "title": title,
                            "text": body,
                            "topics": topics
                        })

df = pd.DataFrame(articles)

print("Finance Articles:", len(df))

df.to_csv("data/finance_reuters_articles.csv", index=False)