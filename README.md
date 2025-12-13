# Crypto Multi-Agent Investment Analyst

This project implements a **multi-agent AI system** for analyzing cryptocurrency data and generating **investment reports with actionable recommendations**.

The system is designed with:
- clear **agent separation of concerns**
- **tool calling** for data retrieval and analysis
- **human-in-the-loop** interaction via a ChatUI interface like ChatGPT
- support for **multiple LLM backends**

---

## 🚀 Getting Started

### Create a Python virtual environment and install dependencies
From the project root directory:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 🤖 LLM Backend Options

This project supports two LLM implementations:
1. Option 1: OpenAI (Default & Recommended) -> Uses OpenAI’s API for LLM reasoning and agent coordination.


Requirements:
- An OpenAI API key
- Active OpenAI billing (minimum $5 credit)
- watch this video to find out how: https://www.youtube.com/watch?v=F10-xRzX7Cg

Setup:

Create a .env file in the project root:
```
OPENAI_API_KEY=your_api_key_here
```

Ensure the .env file is not committed (already ignored via .gitignore).
✅ This is the default configuration and the recommended way to run the project.
