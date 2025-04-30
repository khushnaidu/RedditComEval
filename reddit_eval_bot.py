# Built by a slightly sleep-deprived but determined khush
# RAG + AstraDB + GPT-3.5 = comment rating machine

from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import cassio
from openai import OpenAI
import config

# === setup AstraDB stuff ===
cassio.init(
    database_id=config.ASTRA_DB_ID,
    token=config.ASTRA_DB_APPLICATION_TOKEN
)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# connect to vector store (hosted in AstraDB)
print("Connecting to AstraDB...")
db = AstraDBVectorStore(
    embedding=embedding,
    collection_name=config.COLLECTION_NAME,
    api_endpoint=config.ASTRA_DB_API_ENDPOINT,
    token=config.ASTRA_DB_APPLICATION_TOKEN,
    content_field="text"
)

# === get user query ===
query = input("What do you wanna RAG about? ")

# === search semantically similar comments ===
print("Searching AstraDB for similar examples...")
docs = db.similarity_search(query, k=3)

# manually patch metadata if it doesn't exist
for doc in docs:
    if not hasattr(doc, "metadata") or not isinstance(doc.metadata, dict):
        doc.metadata = {}
    if "rating" not in doc.metadata:
        rating_guess = getattr(doc, "rating", None)
        if rating_guess is not None:
            doc.metadata["rating"] = rating_guess

# build prompt using retrieved docs
context = "\n\n".join([
    f"Reddit Comment Example (Rated {doc.metadata.get('rating', '?')}/100):\n{doc.page_content}"
    for doc in docs
])

prompt = f"""
You are a Reddit content evaluator. Your job is to rate a user's comment on a scale from 1 to 100 based on how interesting or engaging it is, just like the example comments shown below.

Each example shows a Reddit comment and the rating it received. Based on the content and tone of those examples, you should decide how interesting the new comment is, and respond ONLY with a number between 1 and 100 — no words.

Examples:

{context}

Now rate this new Reddit comment:
"{query}"
"""

print("\nSending prompt to GPT...")
client = OpenAI(api_key=config.OPENAI_API_KEY)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You're a witty, informed Reddit user."},
        {"role": "user", "content": prompt}
    ]
)

# print result (straight from GPT's mouth)
print("\nGPT rating:")
print(response.choices[0].message.content)
