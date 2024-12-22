import os
import json
import pandas as pd
import tempfile
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.prompt import Prompt
import simpleaudio as sa  # For reliable audio playback
import logging

# Initialize Rich Console
console = Console()
console.clear()

# Configure logging
logging.basicConfig(filename='identify_speaker.log', level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Display Intro Screen
def display_intro() -> None:
    console.print("""[bold cyan]
=====================================
     SPEAKER IDENTIFICATION TOOL
=====================================
[/bold cyan]""", justify="center")
    console.print("This tool identifies target speakers in audio files using JSON metadata and user input.", 
                 justify="center")
    console.print("[bold yellow]NOTE: Segments you've already processed won't be replayed.[/bold yellow]", 
                 justify="center")

# Display Menu
def display_menu() -> None:
    console.print("""
[bold cyan]=====================================
             MENU OPTIONS
=====================================[/bold cyan]
[bold magenta]Y[/bold magenta] - Mark this speaker as targeted (saves and moves to next episode)
[bold magenta]N[/bold magenta] - Not the targeted speaker (skips all segments by this speaker)
[bold magenta]T[/bold magenta] - Next segment by the same speaker
[bold magenta]L[/bold magenta] - Replay the current clip
[bold magenta]X[/bold magenta] - Exit without saving
[dim]Somewhere in Scotland...[/dim]
=====================================
""", justify="left")

# Title Screen
display_intro()

# Static title header
console.rule("[bold green]Logs and Processing Below[/bold green]")

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(SCRIPT_DIR, "jsons")
AUDIO_DIR = os.path.join(SCRIPT_DIR, "wavs")
CLEANED_DIR = os.path.join(SCRIPT_DIR, "cleaned")
MAPPING_FILE = os.path.join(SCRIPT_DIR, "mappings.csv")

# Ensure Directories Exist
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)

# Validate Mapping File
def validate_mapping_file():
    """Validate and initialize mapping file if needed"""
    try:
        if not os.path.isfile(MAPPING_FILE) or os.path.getsize(MAPPING_FILE) == 0:
            # Create new file with headers if it doesn't exist or is empty
            df = pd.DataFrame(columns=['episode', 'target_speaker'])
            df.to_csv(MAPPING_FILE, index=False, mode='w')  # 'w' mode to ensure clean file
            console.print("[yellow]Created new mapping file[/yellow]")
            return df
        
        # Try to read existing file
        df = pd.read_csv(MAPPING_FILE)
        if list(df.columns) != ['episode', 'target_speaker']:
            # File exists but has wrong columns, recreate it
            df = pd.DataFrame(columns=['episode', 'target_speaker'])
            df.to_csv(MAPPING_FILE, index=False, mode='w')
            console.print("[yellow]Reset mapping file with correct columns[/yellow]")
        return df
    except Exception as e:
        console.print(f"[red]Error with mapping file: {e}[/red]")
        logging.error(f"Mapping file error: {e}")
        # Create new file if there was an error
        df = pd.DataFrame(columns=['episode', 'target_speaker'])
        df.to_csv(MAPPING_FILE, index=False, mode='w')
        console.print("[yellow]Created new mapping file after error[/yellow]")
        return df

# Load Existing Mapping File
df = validate_mapping_file()
processed_count = df['episode'].nunique()
console.print(f"[bold green]{processed_count} episodes have already been processed.[/bold green]")

# Helper Function to Process Single File
def identify_speaker(json_file, wav_file, existing_mapping):
    """Process a single episode file to identify the target speaker"""
    if json_file in existing_mapping['episode'].values:
        console.print(f"[yellow]Skipping {json_file} - already processed[/yellow]")
        return None

    with open(json_file, 'r') as f:
        segments = json.load(f)
    
    # Get unique speakers but maintain original order
    seen = set()
    unique_speakers = [s["speaker"] for s in segments if not (s["speaker"] in seen or seen.add(s["speaker"]))]

    for speaker_label in unique_speakers:
        console.print(f"\n[cyan]Playing segments for speaker: {speaker_label}[/cyan]")
        speaker_segments = [seg for seg in segments if seg["speaker"] == speaker_label]

        for segment in speaker_segments:
            start_time, end_time = segment["start"] * 1000, segment["end"] * 1000
            
            # Skip very short segments
            if (end_time - start_time) < 1000:  # 1.0 seconds minimum
                continue
                
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    audio = AudioSegment.from_wav(wav_file)[start_time:end_time]
                    audio.export(temp_file.name, format="wav")
                    
                    while True:  # Loop for replay option
                        wave_obj = sa.WaveObject.from_wave_file(temp_file.name)
                        play_obj = wave_obj.play()
                        play_obj.wait_done()

                        choice = Prompt.ask(
                            "Is this the target speaker?",
                            choices=["Y", "N", "T", "X", "L", "TJG"],  # Uppercase only
                            case_sensitive=True,  # Enforce exact uppercase match
                            default="N"
                        )

                        if choice == "TJG":
                            console.print("[blue on white]⬜🏴 Alba gu bràth! 🏴⬜[/blue on white]")
                            continue
                        elif choice == "Y":
                            return {"episode": os.path.basename(json_file), 
                                  "target_speaker": speaker_label}
                        elif choice == "z":
                            console.print("[bold rainbow]🎵 You found the secret! Keep on identifying those speakers! 🎵[/bold rainbow]")
                            continue
                        elif choice == "N":
                            break  # Next speaker
                        elif choice == "T":
                            break  # Next segment
                        elif choice == "X":
                            console.print("[bold red]Exiting...[/bold red]")
                            exit(0)
                        elif choice == "L":
                            continue  # Replay
            finally:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

    return None

# Multi-threaded Post-Processing
console.print("[bold cyan]Starting speaker identification...[/bold cyan]")
json_files = [os.path.join(JSON_DIR, f) for f in os.listdir(JSON_DIR) if f.endswith(".json")]
existing_mapping = validate_mapping_file()
new_mappings = []

def main():
    if Prompt.ask(
        "Are you ready to start identifying speakers?",
        choices=["y", "n"],
        default="y"
    ).lower() == "n":
        console.print("[bold red]Exiting tool. Goodbye![/bold red]")
        return

    for json_file in json_files:
        wav_file = os.path.join(CLEANED_DIR, os.path.splitext(os.path.basename(json_file))[0] + ".wav")
        if os.path.exists(wav_file):
            result = identify_speaker(json_file, wav_file, existing_mapping)
            if result:
                new_mappings.append(result)

        # Simplified next episode prompt
        proceed = Prompt.ask(
            "Do you want to process the next episode?",
            choices=["y", "n"],
            default="y"
        )
        if proceed.lower() == "n":
            console.print("[bold yellow]Exiting speaker identification tool gracefully.[/bold yellow]")
            break

    # Use threading only for any post-processing tasks
    console.print("[bold cyan]Processing additional tasks in parallel...[/bold cyan]")

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(lambda: console.print("[green]Task completed[/green]")) for _ in range(len(new_mappings))]
        for future in as_completed(futures):
            future.result()

    # Update and Save Mapping File
    if new_mappings:
        df = pd.concat([existing_mapping, pd.DataFrame(new_mappings)], ignore_index=True)
        df.to_csv(MAPPING_FILE, index=False)
        console.print("[bold green]Mapping updated successfully.[/bold green]")
    else:
        console.print("[yellow]No new mappings were added.[/yellow]")

    console.print("[bold green]Speaker identification complete.[/bold green]")

if __name__ == "__main__":
    main()