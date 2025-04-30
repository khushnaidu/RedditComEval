import json

with open("reddit_comments_rag_with_vectors.json") as f:
    flat_docs = json.load(f)

wrapped_docs = []
for doc in flat_docs:
    wrapped_docs.append({
        "text": doc["text"],
        "vector": doc["vector"],
        "metadata": {
            "rating": doc.get("rating", 0)
        }
    })

with open("reddit_comments_rag_with_metadata.json", "w") as f:
    json.dump(wrapped_docs, f, indent=2)

print("✅ Ready to reupload as metadata-compliant JSON")
