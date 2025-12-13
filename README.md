# Crypto Multi-Agent Investment Analyst

This project implements a **multi-agent AI system** for analyzing cryptocurrency data and generating **investment reports with actionable recommendations**.

The system is designed with:
- clear **agent separation of concerns**
- **tool calling** for data retrieval and analysis
- **human-in-the-loop** interaction via a ChatUI interface like ChatGPT
- support for **multiple LLM backends**

---
![Multi-Agent Architecture](assets/architecture.png)

| ![Architecture]([assets/architecture.png](https://github.com/haidiazaman/crypto-multi-agent-investment-analyst/blob/main/imgs/image.png)) | ![Other Image]([assets/other-image.png](https://github.com/haidiazaman/crypto-multi-agent-investment-analyst/blob/main/imgs/image%20copy.png)) |

---

## 🚀 Getting Started

### Create a Python virtual environment and install dependencies
Use Python 3.12.4

From the project root directory:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 🤖 LLM Backend Options

This project supports two LLM implementations:
#### Option 1: OpenAI (Default & Recommended) 
-> Uses OpenAI’s API for LLM reasoning and agent coordination.


Requirements:
- An OpenAI API key
- Active OpenAI billing (minimum $5 credit)
- watch this video to find out how: https://www.youtube.com/watch?v=F10-xRzX7Cg

Setup:

Create a .env file in the project root and the following line:
```
OPENAI_API_KEY=your_api_key_here
```

Ensure the .env file is not committed (already ignored via .gitignore).

✅ This is the default configuration and the recommended way to run the project.


#### Option 2: Local Ollama Model (Advanced / Not Recommended)
- Uses a locally hosted Ollama model instead of OpenAI.
- Use the OllamaLLM class in src/models/ollama_model.py


## 🧪 Running the Application

Before running any mode, make sure you are in the project root directory and your virtual environment is activated.

### 🐞 Run in Debug / Terminal Mode

This mode runs the full multi-agent pipeline directly in the terminal with verbose output.
```
cd <path-to-project-root>
export PYTHONPATH="."
python src/run_orchestrator_terminal.py
```

### 💬 Run in Chat UI Mode (Streamlit)

This launches an interactive chatbot-style UI for human-in-the-loop interaction.
```
cd <path-to-project-root>
export PYTHONPATH="."
streamlit run src/app.py
```

Once started, Streamlit will display a local URL in the terminal (usually http://localhost:8501).
