# this script take sthe output from preprocessing the video (transcript summary, frame caption pairs) and creates the high quality
# contextual frame descriptions for our vector earch

from claude_utils import (
    create_contextual_frame_description,
    make_claude_transcript_summary,
)
from tqdm import tqdm

import json

with open("./data/all_videos_data.json", "r") as f:
    all_videos_data = json.load(f)


finalized_data = []

for video, data in all_videos_data.items():
    with open(data["transcription"], "r") as f:
        transcript = json.load(f)
    with open(data["frames_and_words"], "r") as f:
        frame_caption_pairs = json.load(f)

    transcript_summary = make_claude_transcript_summary(transcript=transcript)

    print(transcript_summary)

    for i, pair in tqdm(enumerate(frame_caption_pairs)):
        contextual_frame_description = create_contextual_frame_description(
            frame_caption_index=i,
            frame_caption_pairs=frame_caption_pairs,
            transcript_summary=transcript_summary,
        )
        # write out the updated frame caption pairs
        new_pair = {
            "frame_path": pair["frame_path"],
            "words": pair["words"],
            "timestamp": pair["timestamp"],
            "transcript_summary": transcript_summary,
            "contextual_frame_description": contextual_frame_description,
        }
        finalized_data.append(new_pair)

# write out the finalized data
with open("./data/finalized_data.json", "w") as f:
    json.dump(finalized_data, f)
