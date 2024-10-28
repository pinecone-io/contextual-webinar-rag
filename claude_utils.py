import json
import logging
import base64

MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
MAX_TOKENS = 256
system_prompt = '''
You are an expert in the field of AI, and are assistant employees of an AI company on searching over their webinars and 
YouTube videos. You are tasked with, given a bunch of video screencaps, context, and transcriptions, to respond
to the users query and questions for information.

You may be provided several images, text or both as context. You may recieve an image (such as a slide to ask about),
or a text query, or both. You should respond to the user's query or question based on the context provided.

You should provide a response that is informative and helpful to the user, and you should refer back to the source presentations 
where that information came from.

'''

from anthropic import AnthropicBedrock


def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        binary_data = image_file.read()
        base_64_encoded_data = base64.b64encode(binary_data)
        base64_string = base_64_encoded_data.decode('utf-8')
    return base64_string


def ask_claude(img, text):
    client = AnthropicBedrock(
    # Authenticate by either providing the keys below or use the default AWS credential providers, such as
    # using ~/.aws/credentials or the "AWS_SECRET_ACCESS_KEY" and "AWS_ACCESS_KEY_ID" environment variables.
    # Temporary credentials can be used with aws_session_token.
    # Read more at https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp.html.
    # aws_region changes the aws region to which the request is made. By default, we read AWS_REGION,
    # and if that's not present, we default to us-east-1. Note that we do not read ~/.aws/config for the region.
    aws_region="us-east-1")
    if img:
        img_b64 = convert_image_to_base64(img)
        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
            {
                "role": "user", 
                "content": [
                    {"type": "image", "source": 
                        {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64
                        }
                    },
                    {"type": "text", "text": text}
                ]
            }
        ]
        )
    else:
        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": text}]
        )
    return message.content[0].text

def make_claude_transcript_summary(transcript):
    client = AnthropicBedrock(
    aws_region="us-east-1")

    prompt = "Summarize the following transcript, being as concise as possible:"
    message = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt + ": " + transcript}],
        max_tokens=MAX_TOKENS
    )
    return message.content[0].text

def create_contextual_frame_description(frame_caption_index, frame_caption_pairs, transcript_summary, window=60, frame_width=15):
    # frame caption pair will have an image, and a transcript. Window is in seconds
    client = AnthropicBedrock(
    aws_region="us-east-1")

    # gather context, look 4 frame widths before and after. Make sure not to go out of bounds if near beginning or end of video.
    
    surrounding_frames = frame_caption_pairs[max(0, frame_caption_index - 4 * frame_width):frame_caption_index + 1]

    current_frame = frame_caption_pairs[frame_caption_index]

    # summarize past frames
    # removed for now
   #past_frames_summary = make_claude_transcript_summary(" ".join([f["words"] for f in surrounding_frames]))
    meta_prompt = f'''

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
    '''

    rich_summary = ask_claude(img=current_frame["frame_path"], text=meta_prompt)
    return rich_summary


# time the time it takes to run the function
import time

if __name__ == "__main__":
    start = time.time()
    print(system_prompt)
    print(ask_claude("./mlsearch_webinar/frame_0140.png", "What's in this image?"))
    print(f"Time taken: {time.time() - start} seconds")

