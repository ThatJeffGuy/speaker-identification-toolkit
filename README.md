# Speaker Identification Toolkit / SID Toolkit

The SID Toolkit is a collection of Python scripts for extracting, analyzing, and processing audio from video files, with a focus on speaker identification and isolation.

## Scripts Overview

### 1. organize-videos.py
Renames video files by extracting their SxxEyy (Season/Episode) tags.

**Features:**
- Automatically extracts season and episode tags from filenames
- Multi-threaded processing for faster operations
- Graceful error exits - new users to the toolkit will see more descriptive errors!
- Detailed progress reporting with Rich console output

### 2. dataset-creation.py
Extracts English audio tracks from video files using FFmpeg with essential metadata passthrough.

**Features:**
- Identifies and extracts only English audio tracks
- Converts to 16-bit PCM, 44.1kHz mono audio
- Multi-threaded processing with progress tracking
- Strips unnecessary metadata

### 3. diarize-dataset.py
Processes WAV audio files and generates speaker diarization metadata using the pyannote.audio library.

**Features:**
- GPU-accelerated speaker diarization
- Validates and standardizes audio formats
- Generates detailed JSON output with speaker timestamps

### 4. identify-speaker.py
Interactively identifies target speakers in audio files, using the JSON metadata from diarization.

**Features:**
- Interactive audio playback for speaker identification
- Segment navigation and management
- Stores speaker mappings for batch processing

### 5. isolate-trim.py
Isolates and extracts audio segments for specific speakers based on the diarization data and speaker mapping.

**Features:**
- Multi-threaded processing
- Isolates only the target speaker segments
- Creates individual WAV files for each utterance

## Requirements

The scripts require the following dependencies:
- Python 3.6+
- Rich (for console output)
- FFmpeg (for audio extraction)
- pyannote.audio (for speaker diarization)
- pydub (for audio processing)
- pandas (for data handling)
- sounddevice (for audio playback)
- numpy/scipy (for audio processing)

## Installation

1. Clone this repository
2. Install Python requirements: `pip install -r requirements.txt`
3. Install FFmpeg according to your operating system

## Usage Workflow

1. **Organize Videos**: Run `organize-videos.py` to standardize video filenames
2. **Extract Audio**: Run `dataset-creation.py` to extract English audio tracks
3. **Diarize Speakers**: Run `diarize-dataset.py` to identify different speakers
4. **Identify Target Speakers**: Run `identify-speaker.py` to mark speakers of interest
5. **Isolate Audio**: Run `isolate-trim.py` to extract audio segments of target speakers

## Directory Structure

The scripts expect the following directory structure:
```
.
├── dataset-creation.py
├── diarize-dataset.py
├── identify-speaker.py
├── isolate-trim.py
├── organize-videos.py
├── videos/          # Source video files
├── wavs/            # Extracted audio files
├── jsons/           # Speaker diarization data
└── targeted/        # Isolated speaker audio clips
```

## Notes

- For speaker diarization, you need a valid Hugging Face token in `diarize-dataset.py`
- Using good quality audio sources will significantly improve diarization results
- CPU processing is working on all scripts, however it's STRONGLY advised to run the `diarize-dataset.py` on the GPU
- All scripts will automaticly use CUDA utilization where applicable. Sorry AMD.
