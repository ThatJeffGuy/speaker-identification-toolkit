# Speaker Identification Toolkit

This repository provides a comprehensive toolkit for processing audio and video files, with a focus on speaker diarization, speaker identification, audio extraction, and dataset creation. By leveraging tools like `ffmpeg`, `pyannote.audio`, and other Python libraries, the scripts enable efficient and accurate workflows for handling audio data.

## Workflow

### 1. Extract Audio from Videos - STABLE
- Run `dataset-creation.py` to extract English audio tracks from video files.
- Now uses `PyDub` instead of `ffmpeg`, extracts 44100khz PCM-16bit Mono wav files.
- Uses multi-threading for faster, scalable performance.
- Optional Script: `organize-videos.py` will extract Season and Episode info from the file names, and rename them accordingly.
  - This keeps your video files and generated wavs/jsons, uniquenly named like S02E13.wav and S02E13.json, etc.
 
### 1B. Use UVR or a similar vocal isolation project - [UVR Project](https://github.com/Anjok07/ultimatevocalremovergui)
 - Diarizing audio with background noise, music, etc. will result in very poor diarization results, ex. singing from background music will be labeled as a speaker, etc.

### 2. Generate Speaker Diarization Data - STABLE
- Run `diarize-dataset.py` to process the extracted WAV files and produce JSON files containing diarization data.
- Uses `PyDub` instead of `ffmpeg`. Requires a HuggingFace Token.
- Confirms and converts (if necessessary) WAV to SVC/RVC standard, `44100hz PCM-16bit mono`.

### 3. Identify the Target Speaker - BETA RELEASE - ACTIVE PROJECT
- Run `identify-speaker.py` to play audio segments from diarization files and interactively map the target speaker.
- Uses `PyAnnote.audio` for playback, for wider performance and compatibility. `playsound` fallback.

### 4. Isolate and clean-up the Audio - BETA RELEASE
- Run `isolate-trim.py` to extract and trim the target speaker's audio segments, preparing them for dataset creation.
- Uses `PyDub` instead of `ffmpeg` to extract audio segments - while similar, pydub was more compatible than ffmpeg.

## Dependencies

Ensure the following are installed:

- **Python 3.9**
- **ffmpeg**: Install via your system's package manager or from the [official site](https://ffmpeg.org/).

## Configuration

The scripts automatically create necessary directories and pause execution for users to populate them with required data. Ensure the following directory structure is in place:

- **Video Input Directory**: `base-folder/videos`
- **WAV Output Directory**: `base-folder/wavs`
- **JSON Output Directory**: `base-folder/jsons`
- **Speaker Mapping File**: `base-folder/mappings.csv`
- **Processed Speaker Output Directory**: `base-folder/targeted`
