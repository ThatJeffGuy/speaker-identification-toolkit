import os
import json
import wave
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import pandas as pd
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
import logging
from typing import Optional, Dict, List, Set, Any
from dataclasses import dataclass

# Initialize Rich Console
console = Console()
console.clear()

# Configure logging
logging.basicConfig(filename='identify_speaker.log', level=logging.DEBUG, 
                   format='%(asctime)s - %(message)s')

# Constants
AUDIO_TEST_FREQUENCY = 440  # Hz
AUDIO_TEST_DURATION = 0.5   # seconds

@dataclass
class SessionState:
    """Tracks the current session state"""
    df: Optional[pd.DataFrame] = None
    new_mappings: List[Dict] = None
    device_id: Optional[int] = None
    
    def __post_init__(self):
        self.new_mappings = []

class AudioDeviceError(Exception):
    """Raised when there are issues with audio device operations"""
    pass

class FileProcessingError(Exception):
    """Raised when there are issues processing input files"""
    pass

# Initialize global variables
df = None
new_mappings = []

def display_screen(screen_type: str = "main", **kwargs) -> None:
    """Universal display function for all screens
    
    Args:
        screen_type: Type of screen to display ("main", "processing", "status")
        **kwargs: Additional arguments needed for specific screens
    """
    console.clear()
    
    # Common header
    console.print("""[bold cyan]
=====================================
     SPEAKER IDENTIFICATION TOOL
=====================================
[/bold cyan]""", justify="center")
    
    if screen_type == "main":
        console.print("This tool identifies target speakers in audio files using JSON metadata and user input.", 
                     justify="center")
        console.print("[bold yellow]NOTE: Segments you've already processed won't be replayed.[/bold yellow]\n", 
                     justify="center")
    
    # Menu display
    menu_text = """[bold cyan]=====================================
Available Actions:
=====================================[/bold cyan]
[bold magenta]Y[/bold magenta] - Mark as target speaker  [dim](Save and move to next episode)[/dim]
[bold magenta]N[/bold magenta] - Not the target speaker  [dim](Skip all segments by this speaker)[/dim]
[bold magenta]T[/bold magenta] - Next segment           [dim](Move to next segment by same speaker)[/dim]
[bold magenta]L[/bold magenta] - Replay current clip    [dim](Listen to this segment again)[/dim]
[bold magenta]X[/bold magenta] - Exit                   [dim](Exit without saving)[/dim]
====================================="""
    
    if screen_type == "processing":
        data = kwargs.get('data', {})
        episode_name = kwargs.get('episode_name', '')
        current_speaker = kwargs.get('current_speaker', '')
        skipped_speakers = kwargs.get('skipped_speakers', set())
        
        # Calculate statistics
        total_speakers = len({seg['speaker'] for seg in data['segments']})
        identified_speakers = len(skipped_speakers)
        speaker_segments = sum(1 for seg in data['segments'] if seg['speaker'] == current_speaker)
        remaining_speakers = total_speakers - len(skipped_speakers)
        
        # Display processing status
        console.print(f"""
[yellow]Episode:[/yellow] {episode_name}
[yellow]Speakers Progress:[/yellow] {identified_speakers}/{total_speakers} processed
[yellow]Current Speaker:[/yellow] {current_speaker}
[yellow]Speaker Segments:[/yellow] {speaker_segments} total segments
[yellow]Remaining Speakers:[/yellow] {remaining_speakers}
""")
    
    console.print(menu_text)
    
    if screen_type == "main":
        console.print("\n[bold green]Current Processing Status:[/bold green]")
        console.rule()

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(SCRIPT_DIR, "jsons")
AUDIO_DIR = os.path.join(SCRIPT_DIR, "wavs")
MAPPING_FILE = os.path.join(SCRIPT_DIR, "mappings.csv")

# Ensure Directories Exist
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# Validate Mapping File
def validate_mapping_file():
    """Validate and initialize mapping file if needed"""
    global df
    try:
        if not os.path.isfile(MAPPING_FILE) or os.path.getsize(MAPPING_FILE) == 0:
            df = pd.DataFrame(columns=['episode', 'target_speaker'])
            df.to_csv(MAPPING_FILE, index=False, mode='w')
            console.print("[yellow]Created new mapping file[/yellow]")
            return df
        
        df = pd.read_csv(MAPPING_FILE)
        if list(df.columns) != ['episode', 'target_speaker']:
            df = pd.DataFrame(columns=['episode', 'target_speaker'])
            df.to_csv(MAPPING_FILE, index=False, mode='w')
            console.print("[yellow]Reset mapping file with correct columns[/yellow]")
        return df
    except Exception as e:
        console.print(f"[red]Error with mapping file: {e}[/red]")
        logging.error(f"Mapping file error: {e}")
        df = pd.DataFrame(columns=['episode', 'target_speaker'])
        df.to_csv(MAPPING_FILE, index=False, mode='w')
        console.print("[yellow]Created new mapping file after error[/yellow]")
        return df

# Load initial mapping file
df = validate_mapping_file()
processed_count = df['episode'].nunique()
console.print(f"[bold green]{processed_count} episodes have already been processed.[/bold green]")

def save_progress():
    """Save current progress to mapping file"""
    global df, new_mappings
    if new_mappings:
        df = pd.concat([df, pd.DataFrame(new_mappings)], ignore_index=True)
        df.to_csv(MAPPING_FILE, index=False)
        new_mappings = []  # Clear after saving
        return True
    return False

# Menu Constants
MENU_CHOICES = {
    'Y': 'Mark as target speaker',
    'N': 'Not the target speaker',
    'T': 'Next segment by same speaker',
    'L': 'Replay current clip',
    'X': 'Exit'
}

def select_audio_device() -> int:
    """Display available audio devices and let user select one
    
    Returns:
        int: Selected device ID
        
    Raises:
        AudioDeviceError: If no valid output devices are found
    """
    devices = sd.query_devices()
    output_devices = [(i, dev) for i, dev in enumerate(devices) 
                     if dev['max_output_channels'] > 0]
    
    if not output_devices:
        raise AudioDeviceError("No audio output devices found")
    
    # Create a table of output devices
    table = Table(title="Available Audio Output Devices")
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Channels", justify="right", style="yellow")
    table.add_column("Default", style="magenta")
    
    # Filter for output devices
    default_output = sd.default.device[1]
    
    for i, dev in output_devices:
        is_default = "✓" if i == default_output else ""
        table.add_row(
            str(i),
            dev['name'],
            str(dev['max_output_channels']),
            is_default
        )
    
    console.print(table)
    
    try:
        choice = Prompt.ask(
            "\nSelect audio output device",
            choices=[str(i) for i, _ in output_devices],
            default=str(default_output)
        )
        device_id = int(choice)
        
        # Test the selected device
        console.print(f"\n[yellow]Testing audio device: {devices[device_id]['name']}[/yellow]")
        try:
            # Generate test tone with configurable parameters
            sample_rate = int(devices[device_id]['default_samplerate'])
            duration = AUDIO_TEST_DURATION
            test_tone = np.sin(2 * np.pi * AUDIO_TEST_FREQUENCY * 
                               np.arange(int(duration * sample_rate)) / sample_rate)
            sd.play(test_tone, sample_rate, device=device_id)
            sd.wait()
        except Exception as e:
            console.print(f"[red]Error testing audio device: {e}[/red]")
            logging.error(f"Audio device test error: {e}")
            return default_output  # Fallback to default device
        return device_id
    except Exception as e:
        console.print(f"[red]Error selecting audio device: {e}[/red]")
        logging.error(f"Audio device selection error: {e}")
        return default_output  # Fallback to default device

def play_audio_segment(wav_file: str, start_sec: float, end_sec: float, 
                      device_id: int, retry_count: int = 2) -> None:
    """Play a segment of a WAV file
    
    Args:
        wav_file: Path to WAV file
        start_sec: Start time in seconds
        end_sec: End time in seconds
        device_id: Audio device ID
        retry_count: Number of retries on playback failure
    
    Raises:
        AudioDeviceError: If playback fails after retries
    """
    for attempt in range(retry_count):
        try:
            with wave.open(wav_file, 'rb') as wav:
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                frame_rate = wav.getframerate()
                
                start_frame = int(start_sec * frame_rate)
                end_frame = int(end_sec * frame_rate)
                num_frames = end_frame - start_frame
                
                wav.setpos(start_frame)
                audio_data = wav.readframes(num_frames)
                segment = np.frombuffer(audio_data, dtype=np.int16)
                segment = segment.astype(np.float32) / np.iinfo(segment.dtype).max
                
            # Play the audio
            try:
                sd.play(segment, frame_rate, device=device_id)
                sd.wait()  # Wait until audio is done playing
                return
            except sd.PortAudioError as e:
                if attempt == retry_count - 1:
                    raise AudioDeviceError(f"Failed to play audio after {retry_count} attempts") from e
                device_id = select_audio_device()  # Try to get a new device
                
        except Exception as e:
            console.print(f"[red]Error accessing WAV file: {e}[/red]")
            logging.error(f"WAV file error: {e}")

def identify_speaker(json_file: str, wav_file: str, 
                    existing_mapping: pd.DataFrame, 
                    device_id: int) -> Optional[Dict[str, str]]:
    """Process a single episode file and identify speakers
    
    Args:
        json_file: Path to JSON file
        wav_file: Path to WAV file
        existing_mapping: DataFrame with existing mappings
        device_id: Audio device ID
    
    Returns:
        Optional[Dict[str, str]]: Mapping of episode to speaker if found
    
    Raises:
        FileProcessingError: If file processing fails
    """
    episode_name = os.path.splitext(os.path.basename(json_file))[0]
    
    if episode_name in existing_mapping['episode'].values:
        console.print(f"[yellow]Episode {episode_name} already processed, skipping...[/yellow]")
        return None

    try:
        # Load and validate JSON data
        with open(json_file, 'r') as f:
            segments = json.load(f)
            
        # Enhance segment validation
        valid_segments = []
        for s in segments:
            if isinstance(s, dict):
                speaker = s.get('speaker')
                start = s.get('start')
                end = s.get('end')
                text = s.get('text', '[No transcript available]')
                
                if all(x is not None for x in (speaker, start, end)):
                    s['text'] = text  # Ensure text field exists
                    valid_segments.append(s)
        
        if not valid_segments:
            console.print(f"[yellow]No valid segments found in {episode_name}[/yellow]")
            return None
            
        # Process segments by speaker
        speaker_segments = {}
        for s in valid_segments:
            speaker = s['speaker']
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(s)

        # Sort segments within each speaker group by start time
        for speaker in speaker_segments:
            speaker_segments[speaker].sort(key=lambda x: float(x['start']))
        
        # Process speakers in order
        skipped_speakers = set()
        speakers = sorted(speaker_segments.keys())
        
        for current_speaker in speakers:
            if current_speaker in skipped_speakers:
                continue
                
            for segment in speaker_segments[current_speaker]:
                replay = True
                while replay:
                    # Display status with all segments for current speaker
                    display_screen(
                        "processing",
                        episode_name=episode_name,
                        data={'segments': segments},  # Keep original segments for stats
                        current_speaker=current_speaker,
                        skipped_speakers=skipped_speakers
                    )
                    
                    console.print(f"\n[cyan]Current Segment Text:[/cyan]")
                    text = segment.get('text', 'No text available')
                    console.print(f"{text}\n")
                    
                    try:
                        start_time = float(segment['start'])
                        end_time = float(segment['end'])
                        if start_time < end_time:
                            play_audio_segment(wav_file, start_time, end_time, device_id)
                    except (ValueError, TypeError) as e:
                        console.print(f"[red]Error with timestamp values: {e}[/red]")
                        logging.error(f"Timestamp error in {episode_name}: {e}")
                    
                    choice = Prompt.ask(
                        "Your choice",
                        choices=[k.lower() for k in MENU_CHOICES.keys()],
                        default="n"
                    ).upper()
                    
                    if choice == 'Y':
                        return {'episode': episode_name, 'target_speaker': current_speaker}
                    elif choice == 'N':
                        skipped_speakers.add(current_speaker)
                        replay = False
                        break  # Exit segment loop for this speaker
                    elif choice == 'T':
                        replay = False  # Move to next segment
                    elif choice == 'L':
                        continue  # Replay current segment
                    elif choice == 'X':
                        raise KeyboardInterrupt("User requested exit without saving")
                
                if current_speaker in skipped_speakers:
                    break  # Skip remaining segments for this speaker
        
        return None
                
    except Exception as e:
        console.print(f"[red]Error processing {episode_name}: {e}[/red]")
        logging.error(f"Error processing {episode_name}: {e}")
        return None

def load_files():
    """Load and validate input files"""
    json_files = [os.path.join(JSON_DIR, f) for f in os.listdir(JSON_DIR) if f.endswith(".json")]
    wav_files = [os.path.join(AUDIO_DIR, f) for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
    
    if not json_files:
        console.print(f"[red]No JSON files found in {JSON_DIR}[/red]")
        logging.error(f"No JSON files found in {JSON_DIR}")
        return []
    
    # Validate matching wav files exist
    valid_files = []
    for json_file in json_files:
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        wav_file = os.path.join(AUDIO_DIR, f"{base_name}.wav")
        if os.path.exists(wav_file):
            valid_files.append(json_file)
            logging.info(f"Found valid pair: {json_file} -> {wav_file}")
        else:
            console.print(f"[yellow]Warning: No matching WAV file for {base_name}[/yellow]")
            logging.warning(f"No matching WAV file for {json_file}")
    
    return valid_files

def main() -> None:
    """Main application entry point"""
    session = SessionState()
    
    try:
        display_screen("main")
        json_files = load_files()
        
        if not json_files:
            raise FileProcessingError("No valid files to process")
            
        session.device_id = select_audio_device()
        
        if not Prompt.ask(
            f"Found {len(json_files)} files to process. Ready to start?",
            choices=["y", "n"],
            default="y"
        ).lower() == "y":
            console.print("[bold red]Exiting tool. Goodbye![/bold red]")
            return

        try:
            for json_file in json_files:
                try:
                    wav_file = os.path.join(AUDIO_DIR, os.path.splitext(os.path.basename(json_file))[0] + ".wav")
                    result = identify_speaker(json_file, wav_file, df, session.device_id)
                    if result:
                        session.new_mappings.append(result)
                        console.print(f"[green]Mapped {result['episode']} to speaker {result['target_speaker']}[/green]")
                        save_progress()  # Save after each successful mapping
                except KeyboardInterrupt as e:
                    if str(e) == "User requested exit without saving":
                        console.print("\n[yellow]Exiting without saving...[/yellow]")
                        return
                    console.print("\n[yellow]Process interrupted. Exiting without saving...[/yellow]")
                    return

        except KeyboardInterrupt:
            console.print("\n[yellow]Process interrupted. Exiting without saving...[/yellow]")
            return
        
        console.print("[bold green]Speaker identification complete.[/bold green]")

    except (AudioDeviceError, FileProcessingError) as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        logging.error(f"Fatal error: {e}")
        return
    except KeyboardInterrupt:
        if session.new_mappings:
            if Prompt.ask("Save progress before exiting?", choices=["y", "n"]) == "y":
                save_progress()
        console.print("\n[yellow]Process interrupted. Exiting...[/yellow]")
        return

if __name__ == "__main__":
    main()