# Speaker Identification Toolkit

This repository provides a comprehensive toolkit for processing audio and video files, with a focus on speaker diarization, speaker identification, audio extraction, and dataset creation. By leveraging tools like `ffmpeg`, `pyannote.audio`, and other Python libraries, the scripts enable efficient and accurate workflows for handling audio data.

## Workflow

### 1. Extract Audio from Videos
Use `dataset-creation.py` to extract English audio tracks from video files.

### 2. Generate Speaker Diarization Data
Run `diarize-dataset.py` to process the extracted WAV files and produce JSON files containing diarization data.

### 3. Identify Target Speaker
Execute `identify-speaker.py` to play audio segments from diarization files and interactively map the target speaker.

### 4. Isolate and Trim Target Speaker Audio
Use `isolate-trim.py` to extract and trim the target speaker's audio segments, preparing them for dataset creation.

## Dependencies

Ensure the following are installed:

- **Python 3.8+**
- **ffmpeg**: Install via your system's package manager or from the [official site](https://ffmpeg.org/).

## Configuration

The scripts automatically create necessary directories and pause execution for users to populate them with required data. Ensure the following directory structure is in place:

- **Video Input Directory**: `base-folder/videos`
- **WAV Output Directory**: `base-folder/wavs`
- **JSON Output Directory**: `base-folder/jsons`
- **Speaker Mapping File**: `base-folder/scripts/speaker_mapping.csv`
- **Processed Speaker Output Directory**: `base-folder/targeted`

### Notes

- For `diarize-dataset.py`, you can set the Hugging Face API token directly in the script for prompt-free operation. Replace the placeholder value in the following line:
  ```python
  default="your_hugging_face_token"
  ```
- Verify directory paths and ensure all dependencies are installed to avoid runtime issues.

## Extended Description

### 1. `dataset-creation.py`
- **Description**: Extracts English audio tracks from video files and converts them to mono 16-bit PCM WAV format.
- **Key Features**:
  - Processes video files in `.mp4` and `.mkv` formats.
  - Automatically names output files using a sequential naming convention.
- **Dependencies**: `rich`, `ffmpeg`

### 2. `diarize-dataset.py`
- **Description**: Processes WAV files to generate JSON files containing speaker diarization data.
- **Key Features**:
  - Leverages Hugging Face's `pyannote.audio` for diarization.
  - Outputs JSON files with timestamps and speaker labels.
- **Dependencies**: `pyannote.audio`, `rich`

### 3. `identify-speaker.py`
- **Description**: Plays audio segments from diarization files to help users identify and map the target speaker for isolation.
- **Key Features**:
  - Interactive selection of the target speaker.
  - Maintains a CSV file to map diarization files to speakers.
- **Recommendations**:
  - Listen to multiple segments to confirm the target speaker, as segments may contain mixed audio.
  - If a mistake occurs, update the CSV file to correct mappings.
- **Dependencies**: `playsound`, `rich`, `pandas`, `ffmpeg`

### 4. `isolate-trim.py`
- **Description**: Extracts and trims audio segments of the target speaker, splitting them into smaller clips if necessary.
- **Key Features**:
  - Isolates diarized segments for the target speaker.
  - Trims silence and ensures clips are 30 seconds or shorter.
- **Dependencies**: `rich`, `pandas`, `ffmpeg`
