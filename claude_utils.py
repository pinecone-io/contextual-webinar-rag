import json
import logging
import base64
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

boto3_session = boto3.session.Session()
region_name = boto3_session.region_name
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name,
)

system_prompt = '''
You are an expert in the field of AI, and are assistant employees of an AI company on searching over their webinars and 
YouTube videos. You are tasked with, given a bunch of video screencaps, context, and transcriptions, to respond
to the users query and questions for information.

You may be provided several images, text or both as context. You may recieve an image (such as a slide to ask about),
or a text query, or both. You should respond to the user's query or question based on the context provided.

You should provide a response that is informative and helpful to the user, and you should refer back to the source presentations 
where that information came from.

'''


def claude_multimodal_prompt(
    image_path: str = None,
    description: str = None,
    max_tokens: int = 10000,
    model_id: str = 'anthropic.claude-3-sonnet'
):
    """
    Invokes a model with a multimodal prompt.
    Args:
        image_path (str): The path to the image file.
        description (str): The text description.
        max_tokens (int): The maximum number of tokens to generate.
        model_id (str): The model ID to use.
    Returns:
        dict: The response from the model.
    """

    messages = []

    if image_path:
        with open(image_path, "rb") as image_file:
            content_image = base64.b64encode(image_file.read()).decode('utf8')
        message = {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": content_image}}
            ]
        }
        messages.append(message)

    if description:
        if not messages:
            messages.append({"role": "user", "content": []})
        messages[0]["content"].append({"type": "text", "text": description})

    assert messages, "Please provide either an image and/or a text description"

    body = json.dumps({
        "max_tokens": max_tokens,
        "messages": messages
    })

    try:
        response = bedrock_client.invoke_model(
            body=body, modelId=model_id, accept="application/json", contentType="application/json"
        )
        response_body = json.loads(response.get('body').read())
        return response_body
    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        raise RuntimeError(f"A client error occurred: {message}")

# Example usage
if __name__ == "__main__":
    image_path = "/path/to/image"
    description = "What's in this image?"
    response = claude_multimodal_prompt(image_path=image_path, description=description)
    print(json.dumps(response, indent=4))
