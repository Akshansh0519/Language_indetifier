import os
import streamlit as st
from llama_index.llms.gemini import Gemini
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
import ast
import re
from collections import Counter

# Set your Gemini API key here
os.environ["GOOGLE_API_KEY"] = "AIzaSyAL98c1qUXkdf8ewPhL8mnyngmEVfV5ShQ"  # <-- Replace with your Gemini API key

# Configure Gemini LLM
llm = Gemini(model="models/gemini-1.5-pro-latest")

# Example multilingual code-mixed corpus for BM25 retrieval context (expand as needed)
EXAMPLES = [
    "mera friend super irukku",
    "kya aap coffee piyenge",
    "weather today is super hot",
    "naan iniku school pogala",
    "he is a good ladka",
    "sunrise chala bagundi",
    "she is very sundar",
    "let's go to the bazaar kal",
    "yeh weather mast hai",
    "mera bhai bohot intelligent aur smart hai",
    "gracias amigo majha ghar sundar aahe",
    "buenos dias dost andariki salute",
    "bonjour amigo kaise ho",
    "mein casa es tu casa",
    "danke schön dost",
    "ola amigo apni zindagi",
    "ciao bhai kal milte",
    "namaste amigo guten morgen",
    "privet dost bonsoir",
    "hello dost salut amigo",
    "konnichiwa bhai guten tag",
    "ni hao dost buongiorno",
    "sawasdee kap bhai bonjour",
    "shalom dost bonsoir",
    "hej bhai bonjour",
    "merhaba dost buongiorno",
    "salut amigo dhanyavad",
    "hello amigo gracias danke",
    "hej amigo dhanyavad obrigado",
    "hallo bhai merci grazie",
]

def build_bm25(sentences):
    docs = [Document(text=s) for s in sentences]
    splitter = SentenceSplitter(chunk_size=100, chunk_overlap=10)
    nodes = []
    for doc in docs:
        nodes.extend(splitter.get_nodes_from_documents([doc]))
    return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)

retriever = build_bm25(EXAMPLES)

def extract_first_list(s):
    """Extracts the first Python list from a string."""
    m = re.search(r'\[(.|\n)*?\]', s)
    return m.group(0) if m else s

def detect_token_languages(sentence, context_nodes):
    context_txt = "\n".join([node.text for node in context_nodes])
    prompt = f"""
Given the following sentence, output ONLY a Python list of (token, language) tuples.
Possible languages: English, Hindi, Tamil, Telugu, Marathi, Spanish, French, German, Italian, Portuguese, Russian, Japanese, Chinese, Korean, Thai, Hebrew, Turkish, Arabic, Bengali, Urdu, Gujarati, Punjabi, Malayalam, Kannada, Oriya, Assamese, Maithili, Santali, Nepali, Sinhala, Burmese, Vietnamese, Indonesian, Filipino, Dutch, Greek, Czech, Polish, Swedish, Finnish, Danish, Norwegian, Hungarian, Romanian, Bulgarian, Slovak, Lithuanian, Latvian, Estonian, Croatian, Serbian, Slovenian, Albanian, Bosnian, Macedonian, Azerbaijani, Armenian, Georgian, Uzbek, Kazakh, Mongolian, Pashto, Farsi, Kurdish, Somali, Swahili, Zulu, Xhosa, Yoruba, Igbo, Hausa, Amharic, Malagasy, 
    The tokens are words separated by spaces. Detect the language of each token in the sentence and output ONLY a Python list of (token, language) tuples as shown in the format example below.

Sentence: {sentence}

Output format example: [("bonjour", "French"), ("amigo", "Spanish"), ("kaise", "Hindi"), ("ho", "Hindi"), ("hello", "English"), ("dost", "Hindi"), ("salut", "French")]

Your output:
"""
    response = llm.complete(prompt)
    raw = response.text.strip()
    list_str = extract_first_list(raw)
    try:
        pairs = ast.literal_eval(list_str)
        # Ensure it's a list of tuples
        if isinstance(pairs, list) and all(isinstance(t, tuple) and len(t) == 2 for t in pairs):
            return pairs
        else:
            return None
    except Exception:
        return None

def summarize_counts(token_lang_pairs):
    lang_counts = Counter(lang for _, lang in token_lang_pairs)
    summary = ', '.join([f'{count} {lang}' for lang, count in lang_counts.items()])
    return f"This sentence contains {summary} word(s)."

def format_token_table(token_lang_pairs):
    header = "| Token | Language |\n|-------|----------|\n"
    rows = ''.join([f"| {token} | {lang} |\n" for token, lang in token_lang_pairs])
    return header + rows

st.set_page_config(page_title="🧠 Multilingual Token Language Chatbot")
st.title("🧩 Multilingual Language Detector Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Enter any code-mixed sentence (e.g. 'bonjour amigo kaise ho hello dost salut amigo'):")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.spinner("Detecting languages..."):
        # Retrieve similar sentences for context
        retrieved = retriever.retrieve(user_input)
        # Get token-language pairs
        pairs = detect_token_languages(user_input, retrieved)
        if pairs:
            token_table = format_token_table(pairs)
            summary = summarize_counts(pairs)
            output = f"{token_table}\n\n**{summary}**"
        else:
            output = "Could not parse LLM output. Please try again or refine your input."
        st.session_state.messages.append({"role": "assistant", "content": output})
        with st.chat_message("assistant"):
            st.markdown(output)