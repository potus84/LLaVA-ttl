import streamlit as st
import requests
import argparse
import json
import datetime
import os

from llava.conversation import default_conversation
from llava.constants import LOGDIR
from llava.utils import build_logger, violates_moderation, moderation_msg
from pydantic import BaseSettings

class Config(BaseSettings):
    controller_url: str = "http://localhost:10000"
    concurrency_count: int = 10
    model_list_mode: str = "once"
    share: bool = False
    moderate: bool = False
    embed: bool = False

    class Config:
        env_prefix = "LLAVA_"

# Initialize logger
logger = build_logger("streamlit_web_server", "streamlit_web_server.log")


# Helper function to get model list from the controller
@st.cache_resource
def get_model_list(_cfg: Config):
    ret = requests.post(_cfg.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(_cfg.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort()  # Sort the models in alphabetical order
    return models

# This is a placeholder for your chatbot model's response function.  
def get_bot_response(user_input):  
    # You'll need to implement this function based on your chatbot backend.  
    # For example, this function can send the user_input to a model server and return the response.  
    # Here, it simply echoes the user input for demonstration purposes.  
    return f"Bot: Echoing your input for demo purposes - '{user_input}'"

# Define Streamlit application
def main(models):
    st.title("LLaVA: Large Language and Vision Assistant")

    # Model selector
    model_selector = st.selectbox("Choose a model", options=models)


    # Image uploader
    image = st.file_uploader("Upload an image", type=["jpg", "png"])

    # Preprocess mode for non-square images
    # image_process_mode = st.radio("Preprocess for non-square image", ["Crop", "Resize", "Pad", "Default"])

    # Parameters
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    max_output_tokens = st.slider("Max output tokens", min_value=0, max_value=1024, value=512, step=64)

    if 'conversation' not in st.session_state:  
        st.session_state.conversation = []
    # Display conversation history  
    for message_pair in st.session_state.conversation:  
        user_msg, bot_msg = message_pair  
        st.text_area("You", value=user_msg, disabled=True, height=75)  
        st.text_area("Bot", value=bot_msg, disabled=True, height=75)

    
    # Text input
    user_input = st.text_input("Enter text and press ENTER")
    # Send button
    send_button = st.button("Send")
    # Clear history button
    clear_button = st.button("Clear History")

    # Chat history
    # You need to maintain and display the chat history. This can be done using
    # Streamlit's session state or by other means like writing to a file/database.    
    if send_button and user_input:  
        # Get the bot's response  
        bot_response = get_bot_response(user_input)  

        # Update conversation history  
        st.session_state.conversation.append((user_input, bot_response))  

        # Clear the input box after sending the message  
        st.session_state.user_input = ""  

        # Rerun the app to display the updated conversation  
        st.rerun()
    
    if clear_button:
        st.session_state.conversation = []
        st.rerun()
    

    # Implement the chat logic and other functionalities as needed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:10000")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    config = Config()

    models = get_model_list(config)
    main(models)
