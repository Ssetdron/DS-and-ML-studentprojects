import streamlit as st
from singer_chatbot_core import SingerChatbot
from google import genai
import os

# Set up GenAI client
client = genai.Client(api_key=os.getenv("API_KEY"))

# Initialize RAGBot only once
@st.cache_resource
def load_bot():
    bot = SingerChatbot("Singer_4423_EN.pdf", client)
    bot.build_index()
    return bot

bot = load_bot()

# UI
st.title("Chatbot for Singer Heavy Duty Sewing Machine")
query = st.text_input("Ask a question about sewing:")

if query:
    with st.spinner("Thinking..."):
        system_prompt = (
            "Answer only based on the provided context. "
            "If there isn't enough context, say 'I don't know'."
        )
        response = bot.ask(query, system_prompt=system_prompt)
        st.markdown("### Answer")
        st.write(response)
