# 🧠 RAG Rating Bot

This is a Reddit-style comment evaluator powered by:
- LangChain
- AstraDB vector search
- HuggingFace sentence embeddings
- OpenAI GPT-3.5

It uses semantic search to retrieve similar Reddit comments from a vector database and asks GPT to rate new comments based on those examples.

## 🔧 Setup

1. Clone this repo
2. Create a `config.py` using the `config_template.py` as reference
3. Run the bot:

```bash
python3 rag_eval_bot.py
