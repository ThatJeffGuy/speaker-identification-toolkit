# Speaker Identification Toolkit (Speaker Diarization)

This repository contains Python scripts for processing audio and video files, focusing on speaker diarization, speaker identification, audio extraction, and dataset creation. These tools leverage `ffmpeg`, `pyannote.audio`, and other Python libraries for efficient and accurate processing.

## Workflow

1. **Extract Audio from Videos**:
   Use `dataset-creation.py` to extract English audio tracks from video files.

2. **Generate Speaker Diarization Data**:
   Use `diarize-dataset.py` to process the extracted WAV files and generate JSON files containing diarization data.

3. **Identify Target Speaker**:
   Use `identify-speaker.py` to play audio segments from diarization files and map the target speaker interactively.

4. **Isolate and Trim Target Speaker Audio**:
   Run `isolate-trim.py` to extract and trim the target speaker's audio segments for dataset creation.

## Dependencies

Ensure you have the following installed:

- Python 3.8+

- `ffmpeg`:
  Install `ffmpeg` via your system's package manager or [official site](https://ffmpeg.org/).

## Configuration

The script will automatically create the folders it needs, pause for users to populate them with data, or proceed without prompt if they're already populated.

- **Video Input Directory**: `base-folder/videos`
- **WAV Output Directory**: `base-folder/wavs`
- **JSON Output Directory**: `base-folder/jsons`
- **Speaker Mapping File**: `base-folder/scripts/mappings.csv`
- **Processed Speaker Output Directory**: `base-folder/targeted`

**Additionally:**

- For `diarize-dataset.py`, set the Hugging Face API token in the script manually for faster, prompt-free processing. Insert it between the final set of quotes on line 29.

