import logging
import base64
from botocore.exceptions import ClientError
import streamlit as st


MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
MAX_TOKENS = 256
system_prompt = """
You are an expert in the field of AI, and are assistant employees of an AI company on searching over their webinars and 
YouTube videos. You are tasked with, given a bunch of video screencaps, context, and transcriptions, to respond
to the users query and questions for information.

You may be provided several images, text or both as context. You may recieve an image (such as a slide to ask about),
or a text query, or both. You should respond to the user's query or question based on the context provided.

You should provide a response that is informative and helpful to the user, and you should refer back to the source presentations 
where that information came from.

"""


aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
#profile_name=st.secrets["AWS_PROFILE_NAME"]
region_name = st.secrets["AWS_DEFAULT_REGION"]

from anthropic import AnthropicBedrock

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        binary_data = image_file.read()
        base_64_encoded_data = base64.b64encode(binary_data)
        base64_string = base_64_encoded_data.decode("utf-8")
    return base64_string


def format_messages_for_claude(user_query, vdb_response):
    """
    Formats the user's query and the vector database response into a structured message for Claude.

    Args:
        user_query (str): The user's query.
        vdb_response (list): The response from the vector database, containing images and text.

    Returns:
        list: A list of messages formatted for Claude.
    """
    messages = [{"role": "user", "content": []}]
    # add in the first query
    new_content = [{"type": "text", "text": "The user query is: " + user_query}]
    # we alternate between text, image, and text, where we introduce the iamge, then the text, then the next image, and so on.
    # we append three messages at a time, one for the image, one for the text, and one for the next image.

    for item in vdb_response:
        img_b64 = convert_image_to_base64(item["metadata"]["filepath"])
        new_content.extend(
            [
                {
                    "type": "text",
                    "text": "Image: " + item["metadata"]["filepath"],
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                },
                {
                    "type": "text",
                    "text": "Contextual description: "
                    + item["metadata"]["contextual_frame_description"],
                },
                {
                    "type": "text",
                    "text": "Transcript: " + item["metadata"]["transcript"],
                },
            ]
        )
    # reassign
    messages[0]["content"] = new_content
    return messages


def ask_claude_vqa_response(user_query, vdb_response):
    """
    Sends the user's query and the vector database response to Claude and gets a response.

    Args:
        user_query (str): The user's query.
        vdb_response (list): The response from the vector database, containing images and text.

    Returns:
        str: The response from Claude.
    """
    client = AnthropicBedrock(aws_region=region_name, 
                              aws_access_key=aws_access_key_id, 
                              aws_secret_key=aws_secret_access_key)
    messages = format_messages_for_claude(user_query, vdb_response)
    system_prompt = """

You are a friendly assistant helping people interpret their videos at their company.

You will recieve frames of these videos, with descriptions of what has happened in the frames, as well as a user query

Your job is to ingest the images and text, and respond to the user's query or question based on the context provided.

Refer back to the images and text provided to guide the user to the appropriate slide, section, webinar, or talk
where the information they are looking for is located.
    """
    response = client.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS * 10, system=system_prompt, messages=messages
    )
    return response.content[0].text


def ask_claude(img, text):
    # best for one off queries
    client = AnthropicBedrock(aws_region=region_name, 
                              aws_access_key=aws_access_key_id, 
                              aws_secret_key=aws_secret_access_key)
    if img:
        img_b64 = convert_image_to_base64(img)
        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": text},
                    ],
                }
            ],
        )
    else:
        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": text}],
        )
    return message.content[0].text


def make_claude_transcript_summary(transcript):
    client = AnthropicBedrock(aws_region=region_name, 
                              aws_access_key=aws_access_key_id, 
                              aws_secret_key=aws_secret_access_key)

    prompt = "Summarize the following transcript, being as concise as possible:"
    message = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt + ": " + transcript}],
        max_tokens=MAX_TOKENS,
    )
    return message.content[0].text


def create_contextual_frame_description(
    frame_caption_index, frame_caption_pairs, transcript_summary
):
    # frame caption pair will have an image, and a transcript. Window is in seconds
    # gather context, look 4 frame widths before and after. Make sure not to go out of bounds if near beginning or end of video.

    # surrounding_frames = frame_caption_pairs[max(0, frame_caption_index - 4 * frame_width):frame_caption_index + 1]

    current_frame = frame_caption_pairs[frame_caption_index]

    # summarize past frames
    # removed for now
    # past_frames_summary = make_claude_transcript_summary(" ".join([f["words"] for f in surrounding_frames]))
    meta_prompt = f"""

    You are watching a video and trying to explain what has happened in the video using a global summary, some recent context, and the transcript of the current frame.

    The video has been summarized as follows:
    {transcript_summary}

    The current frame's transcript is as follows:
    {current_frame["words"]}

    You also want to provide a description of the current frame based on the context provided.

    Please describe this video snippet using the information above in addition to the frame visual. Explain any diagrams or code or important text that appears on screen,
    especially if the snippet is of a slide or a code snippet. If there are only people in the frame, focus on the transcript and the context provided to describe what has
    been talked about. If a question was asked, and answered, include the question and answer in the description as well.

    Description:
    """

    rich_summary = ask_claude(img=current_frame["frame_path"], text=meta_prompt)
    return rich_summary
