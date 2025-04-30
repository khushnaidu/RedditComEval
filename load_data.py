from datasets import load_dataset
from transformers import pipeline
import json

# Load and sample Reddit comments
comments_ds = load_dataset(
    "SocialGrep/the-reddit-dataset-dataset", "comments")['train']
sampled = comments_ds.select(range(500))

# Set up sentiment model
sentiment_pipe = pipeline("sentiment-analysis")

# Score and prepare data
scored_data = []
for ex in sampled:
    text = ex.get('body', '').strip()
    if len(text) < 10:
        continue
    sentiment = sentiment_pipe(text[:512])[0]
    label = sentiment['label']
    score = sentiment['score']
    rating = int(
        50 + score * 50) if label == 'POSITIVE' else int(50 - score * 50)
    scored_data.append({"text": text, "rating": rating})

# Save to JSON
with open("reddit_comments_rag.json", "w") as f:
    json.dump(scored_data, f, indent=2)

print("reddit_comments_rag.json saved.")
