import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from backend.rag_engine import SingerChatbot
from google import genai
import re
from PIL import Image
from dotenv import load_dotenv

load_dotenv() 

client = genai.Client(api_key=os.getenv("API_KEY"))

@st.cache_resource
def load_bot():
    bot = SingerChatbot("Singer_4423_EN.pdf", client)
    bot.build_index()
    return bot

bot = load_bot()

def render_response_with_images(response_text, image_folder="chatbot/manual_images"):
    st.markdown("### Answer")

    intro_line = "Let Grandma Berta show you how it's done! 🧵😊\n\n"
    figure_refs = re.findall(r"(?:Fig(?:ure)?\.?\s*)(\d+)", response_text)
    shown = set()

    st.write(intro_line + response_text)

    for fig_num in figure_refs:
        if fig_num in shown:
            continue 
        shown.add(fig_num)
        for ext in ["jpg", "png", "jpeg"]:
            image_path = os.path.join(image_folder, f"fig{fig_num}.{ext}")
            if os.path.exists(image_path):
                st.image(Image.open(image_path), caption=f"Figure {fig_num}", use_column_width=True)
                break  

st.title("Berta the Bot 🧵🤖🪡")
query = st.text_input("Here to answer all your stitching questions:")

if query:
    with st.spinner("Thinking..."):
        system_prompt = (
                "You're Berta the Bot — a wise, quirky, and warm grandma who knows everything about sewing, "
                "especially the Singer 4423 machine. You speak in a casual, grandmotherly tone with charm, a pinch of sass, "
                "and the occasional sewing pun. Use analogies from your own experience, and if you don’t know something, say something like: "
                "'Well honey, that's a tough needle to crack, I’d need to dig a little deeper in my pattern drawer for that one.' "
                "Only answer based on the provided context."
        )
        response_text, pages = bot.ask(query, system_prompt=system_prompt)
        render_response_with_images(response_text, image_folder="data/manual_images")

        st.markdown("### Related Images from Manual")
        for page in pages:
            image_filename = f"page-{page}.jpg" 
            image_path = os.path.join("data/manual_images", image_filename)
            if os.path.exists(image_path):
                st.image(image_path, caption=f"Page {page}")

