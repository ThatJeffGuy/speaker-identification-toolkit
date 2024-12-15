# Speaker Identification Toolkit

## Overview

This toolkit provides a comprehensive suite of scripts for audio processing and speaker identification tasks. It includes features like video-to-audio conversion, speaker diarization, speaker isolation, and audio trimming.

## Scripts

### 1. Dataset Creation (Video to Audio Extraction)

- **Functionality:** Extracts audio tracks from video files.
- **Default Directories:**
  - Input: `videos`
  - Output: `wavs`
- **Behavior:**
  - Auto-creates `videos` and `wavs` directories in the script folder.
  - Pauses execution to allow the user to populate the `videos` directory before processing.
  - Processes all `.mkv` and `.mp4` files in the `videos` directory.
  - Will create WAV's for use with model training: mono, pcm_s16le

### 2. Speaker Diarization

- **Functionality:** Processes audio files and generates speaker diarization metadata.
- **Default Directories:**
  - Input: `wavs`
  - Output: `jsons`
- **Behavior:**
  - Auto-creates `wavs` and `jsons` directories in the script folder.
  - Includes a prompt for the Hugging Face token with validation.
  - Pauses execution to allow the user to populate the `wavs` directory.
  - Generates diarization metadata in `.json` format for all `.wav` files in the `wavs` directory.

### 3. Speaker Isolation and Trimming

- **Functionality:** Isolates and trims audio segments based on speaker diarization metadata.
- **Default Directories:**
  - Input JSON: `jsons`
  - Input WAV: `wavs`
  - Output: `targeted`
- **Behavior:**
  - Auto-creates `jsons`, `wavs`, and `targeted` directories in the script folder.
  - Pauses execution to allow the user to populate the `jsons` and `wavs` directories.
  - Processes each `.json` file and its corresponding `.wav` file to extract and trim speaker-specific segments.
  - Outputs trimmed audio clips in the `targeted` directory.

## Common Features

- All scripts auto-create necessary directories in the same folder as the script.
- Scripts pause execution at key points to allow the user to populate input directories.
- You can manually create the directories in advance and place media in it, for quicker processing.
- Rich console output for enhanced user interaction and error handling.

## Hugging Face Token

- Required for the Speaker Diarization script.
- Token validation is performed before processing.
- Update your token in the script or provide it at runtime when prompted.

## Notes and Tips

- Output files will be saved in the corresponding output directories, which are automatically created if they do not exist.
- When processing multi-speakers in multi-episode files, like a TV series spanning multiple seasons, it is best practice to listen to a few segments of the speaker before confirming.
- If you make a mistake, stop the script and remove the bottom most line in your mappings CSV; the script will re-process it on next run.
