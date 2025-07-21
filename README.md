# 🧠 Multilingual Language Detector Chatbot

An intelligent chatbot that detects the **language of each token** (word) in code-mixed or multilingual sentences using Google's **Gemini 1.5 Pro** and a custom **BM25 context retriever**. Built with Streamlit for an interactive interface.

---

## 🚀 Demo

Paste any sentence like:  **bonjour amigo kaise ho hello dost salut amigo**


And get this as output:

| Token     | Language |
|-----------|----------|
| bonjour   | French   |
| amigo     | Spanish  |
| kaise     | Hindi    |
| ho        | Hindi    |
| hello     | English  |
| dost      | Hindi    |
| salut     | French   |

**Summary**: This sentence contains 2 French, 2 Hindi, 1 Spanish, and 1 English word(s).

---

## 📌 Features

- 🔤 **Word-level language detection** for over 70+ languages
- 🌐 Handles **code-mixed**, multilingual, and hybrid text
- 🤖 Uses **Gemini 1.5 Pro** as the backend LLM
- 🧠 Context-enhanced prompting with **BM25 retrieval**
- ⚡ Clean, responsive UI with **Streamlit**
- 🧪 Built-in sample corpus for better generalization

---

## 🧰 Tech Stack

| Tool           | Usage                          |
|----------------|---------------------------------|
| `Streamlit`    | Web app interface               |
| `Gemini LLM`   | Token-wise language prediction  |
| `BM25Retriever`| Context sentence retrieval      |
| `LlamaIndex`   | Integration and node parsing    |
| `Regex + AST`  | Robust parsing of LLM output    |

---

## 🔧 Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/Akshansh0519/Language_indetifier.git
   cd Language_indetifier
   
2.**Install dependencies**
pip install -r requirements.txt


3.**Add your Gemini API key**
In your terminal or .env:
export GOOGLE_API_KEY="Your_API_Key"


4.**Run the app**
streamlit run app.py
