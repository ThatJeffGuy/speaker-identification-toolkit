# Speaker Identification Toolkit (SID Toolkit)

The SID Toolkit is a collection of Python scripts for extracting, analyzing, and processing audio from video files, with a focus on speaker identification and isolation. It helps you extract and organize audio segments of specific speakers from video content.

## Installation

1. Clone this repository
2. Create VENV, Conda, or a similar isolated environment
3. Install requirements: `pip install -r requirements.txt` - note: `ffmpeg` required, installed separately

## Requirements

The scripts require the following dependencies:
- Python 3.12+ (recommended for best compatibility)
- Rich (for console output)
- FFmpeg (for audio extraction)
- pyannote.audio (for speaker diarization)
- pydub (for audio processing)
- pandas (for data handling)
- sounddevice (for audio playback)
- numpy/scipy (for audio processing)
- CUDA-compatible GPU (recommended for diarization)

## Usage Workflow
1. **Organize Videos**: Run `organize-videos.py` to standardize and organize filenames
2. **Extract Audio**: Run `dataset-creation.py` to extract English audio tracks (creates WAVs)
3. **Diarize Speakers**: Run `diarize-dataset.py` to document timestamps of all speakers (creates JSONs)
4. **Identify Target Speakers**: Run `identify-speaker.py` to mark your targeted speaker
5. **Isolate Audio**: Run `isolate-trim.py` to automatically extract all segments of your targeted speaker (exports to `targeted` directory)

## Directory Structure
The scripts expect the following directory structure:
```
## Notes

- For speaker diarization, you need a valid Hugging Face token in `diarize-dataset.py`
- Using good quality audio sources will significantly improve diarization results
- CPU processing works for all scripts, but it's STRONGLY advised to run `diarize-dataset.py` on a GPU
- All scripts will automatically use CUDA acceleration where applicable (AMD GPUs not supported)
- The filenames between WAV and JSON files must match exactly for the scripts to work properly
