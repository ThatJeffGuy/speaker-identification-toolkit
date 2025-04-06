# Speaker Identification Toolkit / SID Toolkit

The SID Toolkit is a collection of Python scripts for extracting, analyzing, and processing audio from video files, with a focus on speaker identification and isolation.

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

1. **Organize Videos**: Run `organize-videos.py` to standardize video filenames.
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
- Once at the `diarize-dataset.py` stage, note that filenames must match, JSON and WAV.
