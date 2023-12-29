import streamlit as st
import requests
import argparse
import json
from PIL import Image
import hashlib
import base64
from io import BytesIO
from PIL import Image

from llava.conversation import default_conversation, SeparatorStyle
from llava.constants import LOGDIR
from llava.utils import build_logger, violates_moderation, moderation_msg
from llava.eval.run_llava import eval_model
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    controller_url: str = "http://localhost:10000"
    concurrency_count: int = 10
    model_list_mode: str = "once"
    share: bool = False
    moderate: bool = False
    embed: bool = False


# Initialize logger
logger = build_logger("streamlit_web_server", "streamlit_web_server.log")


# Helper function to get model list from the controller
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


# Define a function to get the worker address from the controller
def get_worker_address(controller_url, model_name):
    response = requests.post(
        controller_url + "/get_worker_address", json={"model": model_name}
    )
    return response.json()["address"]


def expand2square(pil_img, background_color=(122, 116, 104)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(
            pil_img.mode,
            (height, height),
            background_color,
        )
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


# Function to handle images
def get_images(source_images, return_pil=False, image_process_mode="Default"):
    images = []
    for i, image in enumerate(source_images):
        if image_process_mode == "Pad":
            image = expand2square(image)
        elif image_process_mode in ["Default", "Crop"]:
            pass
        elif image_process_mode == "Resize":
            image = image.resize((336, 336))
        else:
            raise ValueError(
                f"Invalid image_process_mode: {image_process_mode}"
            )
        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        max_len, min_len = 800, 400
        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        W, H = image.size
        if longest_edge != max(image.size):
            if H > W:
                H, W = longest_edge, shortest_edge
            else:
                H, W = shortest_edge, longest_edge
            image = image.resize((W, H))
        if return_pil:
            images.append(image)
        else:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            images.append(img_b64_str)
    return images


# Define a function to send the chat message and get the response
def send_chat(
    state,
    prompt,
    model_selector,
    temperature,
    top_p,
    max_new_tokens,
    controller_url,
):
    worker_addr = get_worker_address(controller_url, model_selector)

    if not worker_addr:
        st.error("No available worker.")

    # Assuming state has a method 'get_images' to get PIL images
    all_images = get_images(state.images)
    # all_image_hash = [
    #     hashlib.md5(image.tobytes()).hexdigest() for image in all_images
    # ]

    pload = {
        "model": model_selector,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": "###",
        "images": all_images,
    }

    try:
        response = requests.post(
            worker_addr + "/worker_generate_stream",
            headers={"User-Agent": "LLaVA Client"},
            json=pload,
            stream=True,
            timeout=10,
        )
        return response
    except Exception as e:
        st.error("Server error.", e)


# Inference offline
def send_chat_offline(
    model,
    tokenizer,
    image_processor,
    prompt,
    file_name,
    temperature=0.2,
    top_p=0.7,
    max_new_tokens=512,
):
    args = type(
        "Args",
        (),
        {
            "query": prompt,
            "conv_mode": None,
            "image_file": file_name,
            "sep": ",",
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "num_beams": 1,
        },
    )()
    text = eval_model(args, tokenizer, model, image_processor)
    return text


# Define Streamlit application
def main(
    config: Config,
    model_list,
    # model, tokenizer, image_processor, context_len
):
    st.title("LLaVA: Large Language and Vision Assistant")

    # Model selector
    model_selector = st.selectbox("Choose a model", options=model_list)

    # Image uploader
    image_file = st.file_uploader("Upload an image", type=["jpg", "png"])

    if image_file is not None:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Uploaded Image")
        st.session_state.images = [image]

    # Preprocess mode for non-square images
    # image_process_mode = st.radio("Preprocess for non-square image", ["Crop", "Resize", "Pad", "Default"])

    # Parameters
    # Create an expander with a label
    expander = st.expander("Model parameters")
    temperature = expander.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1
    )
    top_p = expander.slider(
        "Top P", min_value=0.0, max_value=1.0, value=0.7, step=0.1
    )
    max_output_tokens = expander.slider(
        "Max output tokens", min_value=0, max_value=1024, value=512, step=64
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # display chat messages from history on app rerun
    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # accept user input
    if prompt := st.chat_input("What would you like to know?"):
        # add user message to chat history
        st.session_state.conversation.append(
            {"role": "user", "content": prompt}
        )
        # display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # full_response = get_bot_response(prompt)
            # full_response = send_chat_offline(
            #     model,
            #     tokenizer,
            #     image_processor,
            #     prompt,
            #     image_file.name,
            #     temperature,
            #     top_p,
            #     max_output_tokens,
            # )
            response = send_chat(
                st.session_state,
                prompt,
                model_selector,
                temperature,
                top_p,
                max_output_tokens,
                config.controller_url,
            )

            for chunk in response.iter_lines(
                decode_unicode=False, delimiter=b"\0"
            ):
                if chunk:
                    data = json.loads(chunk.decode())
                    if data["error_code"] == 0:
                        output = data["text"][len(prompt) :].strip()
                    else:
                        output = (
                            data["text"]
                            + f" (error_code: {data['error_code']})"
                        )
                    full_response += output
                    message_placeholder.markdown(output + "▌")

            message_placeholder.markdown(full_response)
        st.session_state.conversation.append(
            {"role": "assistant", "content": full_response}
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--controller-url", type=str, default="http://localhost:10000"
    )
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument(
        "--model-list-mode",
        type=str,
        default="once",
        choices=["once", "reload"],
    )
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    config = Config()

    model_list = get_model_list(config)
    #model_list = ["liuhaotian/llava-v1.5-13b"]
    # model, tokenizer, image_processor, context_len = load_pretrained_model(
    #     model_path="liuhaotian/llava-v1.5-13b",
    #     load_4bit=True,
    #     model_base=None,
    #     model_name="liuhaotian/llava-v1.5-13b",
    # )
    main(config, model_list)
