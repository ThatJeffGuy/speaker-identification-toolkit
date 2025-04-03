# SID Toolkit - Speaker Identification Toolkit

This collection of tools provides a complete pipeline for extracting, processing, and organizing audio from video files, with a focus on speaker identification and isolation.

## Overview

The SID Toolkit consists of five main scripts that should be run in sequence:

1. **organize-videos.py** - Renames video files by extracting Season/Episode tags
2. **dataset-creation.py** - Extracts English audio tracks from videos
3. **diarize-dataset.py** - Performs speaker diarization on the WAV files
4. **identify-speaker.py** - Interactive tool to identify target speakers
5. **isolate-trim.py** - Isolates and extracts audio segments from the identified speakers

## Requirements

Before running these scripts, make sure you have the following installed:

- Python 3.6 or higher
- FFmpeg (for audio extraction)
- Required Python packages: 
  ```
  pip install rich pydub scipy pandas numpy sounddevice pyannote.audio torch
  ```
- Hugging Face account with access to pyannote/speaker-diarization model

## SID Toolkit Steps

### 1. Organize Videos

```
python organize-videos.py
```

This script renames video files in the `videos` directory by extracting their SxxEyy (Season/Episode) tags.

**Features:**
- Preserves original file extensions
- Multi-threaded processing for faster renaming
- Detailed reporting of renamed, skipped, and failed files

### 2. Extract Audio

```
python dataset-creation.py
```

This script extracts English audio tracks from the video files in the `videos` directory and saves them as WAV files in the `wavs` directory.

**Features:**
- Automatically detects English audio tracks
- Converts to mono 16-bit PCM at 44.1kHz for optimal processing
- Multi-threaded processing for faster extraction

### 3. Diarization

```
python diarize-dataset.py
```

This script performs speaker diarization on the WAV files in the `wavs` directory and saves the results as JSON files in the `jsons` directory.

**Features:**
- Uses pyannote.audio's state-of-the-art speaker diarization model
- GPU acceleration support for faster processing
- Detailed progress reporting

### 4. Speaker Identification

```
python identify-speaker.py
```

This interactive tool helps you identify specific speakers in the audio files by playing segments and letting you mark which speaker you want to target.

**Features:**
- Plays audio segments for each detected speaker
- Interactive interface for marking target speakers
- Saves speaker mappings to `mappings.csv`

### 5. Speaker Isolation

```
python isolate-trim.py
```

This script isolates and extracts audio segments from the identified speakers based on the mappings created in the previous step.

**Features:**
- Multi-threaded processing for faster extraction
- Creates clean, isolated audio files for each speaker segment
- Organizes files with clear naming conventions (filename_speaker_starttime-endtime.wav)

## Directory Structure

The SID Toolkit creates and uses the following directory structure:

```
./
├── videos/          # Original video files
├── wavs/            # Extracted WAV audio files
├── jsons/           # Speaker diarization JSON files
├── targeted/        # Isolated speaker audio segments
├── mappings.csv     # Speaker identification mappings
└── scripts/         # Pipeline scripts
```

## Tips for Best Results

1. **Use high-quality source videos** for better audio extraction.
2. **Run audio through vocal isolators** before diarization for cleaner speaker detection.
3. **Listen to multiple segments per speaker** to ensure correct speaker identification.
4. **Confirm speaker identity across episodes** for consistency.

## Troubleshooting

- **Missing FFmpeg**: Install FFmpeg and ensure it's in your system PATH.
- **Hugging Face token issues**: Update the `HF_TOKEN` variable in `diarize-dataset.py` with your valid token.
- **Audio playback problems**: Ensure your system has a working audio output device.
- **GPU acceleration**: If available, use GPU acceleration by installing CUDA for significantly faster processing.

## Advanced Customization

Each script contains configurable parameters at the top of the file that can be adjusted for your specific needs:

- Change minimum/maximum clip lengths
- Adjust overlap durations
- Modify audio sample rates and formats
- Configure threading behavior

## License

This project is open-source and available for personal and commercial use.
