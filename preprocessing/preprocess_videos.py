import ffmpeg
from transformers import pipeline
import os
import torch
import json 
from config import videos_dir, data_dir

# Important Globals
## seconds to walk over the videos for
INTERVAL = 45


# Step 1: Transcribe Video
def transcribe_video(video_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny",device=device)
    transcription = transcriber(video_path, return_timestamps="word")
    return transcription

# Step 2: Extract Frames and Pair with Dialogue
def extract_frames(frames_output_path, video_path, interval):

    # output dir will be named after the video file
    # this grabs just the name of the video file without the extension
    output_dir = os.path.splitext(os.path.basename(video_path))[0]
    # this is where the frames will be stored, under the name of the video in a folder
    frame_output = os.path.join(frames_output_path, output_dir)
    try:
        os.mkdir(frame_output)
    except FileExistsError:
        print(f"Folder already exists for videofile {video_path}")
        print("Please delete it to start fresh or ensure it is empty.")
    
    # Use ffmpeg to extract frames
    (
        ffmpeg
        .input(video_path)
        .filter('fps', fps=1/interval)
        # this sets the output to be a frame every Interval seconds
        .output(f'{frame_output}/frame_%04d.png')
        .run()
    )
    
    # Collect the frame file paths
    frame_files = sorted([os.path.join(frame_output, f) for f in os.listdir(frame_output) if f.endswith('.png')])
    # create a dictionary of frame file paths and their corresponding timestamps according to interval and video length

    frames_and_intervals = []
    interval_start = 0
    for i, frame in enumerate(frame_files):
        single_frame = {
            "frame_path": frame,
            "timestamp": (interval_start, interval_start + (interval))
        }
        frames_and_intervals.append(single_frame)
        interval_start+=interval
    
    # get video duration and make the last frame the same as the duration
    video_duration = ffmpeg.probe(video_path)['format']['duration']
    final_frame_start_time = frames_and_intervals[-1]["timestamp"][0]
    frames_and_intervals[-1]["timestamp"] = (final_frame_start_time, float(video_duration))
    return frames_and_intervals

def assign_words_to_frames(transcription, frames, interval=INTERVAL):
    # given transcription word chunks and frames that occur w.r.t interval, assign words to frames
    # Transcription will be on word level, so for each frame, we find the words that occur in the time interval of the frame
    # and assign them to the frame

    frames_and_words = []

    for f in frames:
        frame_start, frame_end = f["timestamp"][0], f["timestamp"][1]
        
        # filter for words that fall in frame
        words_in_frame = list(filter(lambda w: w['timestamp'][0] > frame_start and w['timestamp'][1] < frame_end, transcription["chunks"]))
        words = [w['text'] for w in words_in_frame]
        words= "".join(words)
        single_frame = {
            "frame_path": f["frame_path"],
            "words": words,
            "timestamp": f["timestamp"]
        }
        frames_and_words.append(single_frame)
    return frames_and_words

def align_frames_with_dialogue(frames, transcription):
    #Pinecone for t
    dialogue_frames = []
    for frame, chunk in zip(frames, transcription["chunks"]):
        dialogue_frames.append({
            "frame": frame,
            "dialogue": chunk["text"],
            "metadata": {
                "timestamp": chunk["timestamp"]
            }
        })
    return dialogue_frames


if __name__ == "__main__":
    # read in video files from data directory
    video_files = os.listdir(videos_dir)
    print(video_files)
    # add root dir to video files
    video_files = [os.path.join(videos_dir, f) for f in video_files]

    all_videos_data = {}
    transcriptions_dir = os.path.join(data_dir, "transcriptions")
    frames_and_words_dir = os.path.join(data_dir, "frames_and_words")
    frames_dir = os.path.join(data_dir, "frames")
    # folder setup
    try:
        os.mkdir(transcriptions_dir)
        os.mkdir(frames_and_words_dir)
        os.mkdir(frames_dir)
    except FileExistsError:
        print("Folders already exist. Please delete them to start fresh or ensure they are empty.")

    for video_path in video_files:
        transcription = transcribe_video(video_path)

        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        
        # Write transcription out as json
        transcription_filename = os.path.join(transcriptions_dir, video_filename + "_transcription.json")
        with open(transcription_filename, "w") as f:
            json.dump(transcription["text"], f)
        
        frames = extract_frames(frames_dir, video_path, INTERVAL)
        frames_and_words = assign_words_to_frames(transcription, frames)
        
        frames_and_words_filename = os.path.join(frames_and_words_dir, video_filename + "_frames_and_words.json")
        with open(frames_and_words_filename, "w") as f:
            json.dump(frames_and_words, f)
        
        all_videos_data[video_filename] = {
            "transcription": transcription_filename,
            "frames_and_words": frames_and_words_filename
        }

    # Optionally, write all_videos_data to a summary file
    all_videos_data_path = data_dir / "all_videos_data.json"
    with open(all_videos_data_path, "w") as f:
        json.dump(all_videos_data, f)


    