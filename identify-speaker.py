import os
import json
import pandas as pd
import numpy as np
from scipy.io import wavfile
from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from rich.table import Table
import sounddevice as sd

# Initialize Rich Console with a fixed width
CONSOLE_WIDTH = 50
console = Console(width=CONSOLE_WIDTH)

# Clear the terminal screen
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

# Constants
SEPARATOR = "=" * CONSOLE_WIDTH

# Title Screen
def print_title():
    console.print(Panel(
        Align("[bold cyan]SPEAKER IDENTIFICATION TOOL[/bold cyan]", "center"),
        border_style="cyan",
        width=CONSOLE_WIDTH
    ))
    
    note_text = (
        "This tool interactively identifies a targeted "
        "speaker in an audio file, using JSON metadata.\n\n"
        "[italic yellow]NOTE: For the best results, run your audio "
        "through UVR or a similar isolation project. "
        "If you don't, groans, machines wirring, "
        "background music vocals among other things "
        "will get marked as a valid speaker.[/italic yellow]"
    )
    
    console.print(Panel(
        Align(note_text, "center"),
        border_style="blue",
        width=CONSOLE_WIDTH
    ))

# Menu Display
def print_menu():
    menu_items = [
        "[bold magenta]Y[/bold magenta] - Confirm that this is the Targeted Speaker",
        "[bold magenta]N[/bold magenta] - Mark this as NOT the Targeted Speaker",
        "[bold magenta]A[/bold magenta] - Listen to the same clip again",
        "[bold magenta]U[/bold magenta] - Move to the Next Segment for the Same Speaker",
        "[bold magenta]X[/bold magenta] - Stop and Continue Processing Files Later"
    ]
    
    menu_text = "\n".join(menu_items)
    
    console.print(Panel(
        Align("[bold cyan]MENU OPTIONS[/bold cyan]", "center"),
        border_style="cyan",
        width=CONSOLE_WIDTH
    ))
    
    console.print(Panel(
        menu_text,
        border_style="yellow",
        width=CONSOLE_WIDTH
    ))

# Display file and speaker status
def print_status(wav_filename, speaker, num_speakers, speakers_checked, segment, is_repeating):
    status_table = Table(width=CONSOLE_WIDTH-4, box=None, show_header=False)
    status_table.add_column("Key", style="cyan")
    status_table.add_column("Value")
    
    status_table.add_row("File", wav_filename)
    
    if is_repeating:
        status_table.add_row(
            "Speaker", 
            f"[cyan]{speaker}[/cyan] ([cyan]{speakers_checked}/{num_speakers}[/cyan]) [bold red](REPEATING)[/bold red]"
        )
    else:
        status_table.add_row(
            "Speaker", 
            f"[cyan]{speaker}[/cyan] ([cyan]{speakers_checked}/{num_speakers}[/cyan])"
        )
    
    status_table.add_row(
        "Segment", 
        f"[cyan]{segment['start']:.2f}s - {segment['end']:.2f}s[/cyan]"
    )
    
    console.print(Panel(
        Align("[bold cyan]CURRENT STATUS[/bold cyan]", "center"),
        border_style="cyan",
        width=CONSOLE_WIDTH
    ))
    
    console.print(Panel(
        status_table,
        border_style="blue",
        width=CONSOLE_WIDTH
    ))

# Prompt User to Start
def prompt_start():
    console.print(Panel(
        Align("[bold yellow]Are you ready to start identifying speakers?[/bold yellow]", "center"),
        border_style="yellow",
        width=CONSOLE_WIDTH
    ))
    
    start_prompt = console.input("\n(y/n): ").strip().lower()
    while start_prompt not in ["y", "n"]:
        console.print(Panel(
            Align("[bold red]Invalid input. Please enter 'y' or 'n'[/bold red]", "center"),
            border_style="red",
            width=CONSOLE_WIDTH
        ))
        start_prompt = console.input("\n(y/n): ").strip().lower()
    
    if start_prompt == "n":
        console.print(Panel(
            Align("[bold red]Exiting tool. Goodbye![/bold red]", "center"),
            border_style="red",
            width=CONSOLE_WIDTH
        ))
        exit(0)

# Play audio data
def play_audio(audio_data, sample_rate):
    """Plays the audio data using sounddevice."""
    try:
        # Convert to float32 if needed (sounddevice works better with float)
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            # Normalize if int type was converted to float
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / 32768.0  # normalize for 16-bit audio
        
        sd.play(audio_data, sample_rate)
        sd.wait()
    except Exception as e:
        console.print(Panel(
            Align(f"[bold red]Error playing audio: {e}[/bold red]", "center"),
            border_style="red",
            width=CONSOLE_WIDTH
        ))
        # Brief pause to show the error
        import time
        time.sleep(2)

# Get user decision
def get_user_decision():
    console.print(Panel(
        Align("[bold yellow]Is this the targeted speaker?[/bold yellow]", "center"),
        border_style="yellow",
        width=CONSOLE_WIDTH
    ))
    return console.input("\n(Y/N/A/U/X): ").strip().lower()

# Show status message
def show_status_message(message, style="green"):
    console.print(Panel(
        Align(message, "center"),
        border_style=style,
        width=CONSOLE_WIDTH
    ))

# Define Paths Relative to Script Location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(SCRIPT_DIR, "jsons")
AUDIO_DIR = os.path.join(SCRIPT_DIR, "wavs")
MAPPING_FILE = os.path.join(SCRIPT_DIR, "mappings.csv")
PROGRESS_FILE = os.path.join(SCRIPT_DIR, ".progress")

# Ensure Directories Exist
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# Parameters for processing
MIN_CLIP_LENGTH = 1  # in seconds (minimum clip length)
MAX_CLIP_LENGTH = 5  # in seconds (maximum clip length)
OVERLAP_DURATION = 0.5  # Overlap duration in seconds

# Mappings.csv File Handling
def load_or_create_mapping_file():
    """Load existing mapping file or create a new one if it doesn't exist."""
    if not os.path.exists(MAPPING_FILE):
        show_status_message(
            f"Mapping file not found. Creating a new one at {MAPPING_FILE}...",
            style="yellow"
        )
        df_empty = pd.DataFrame(columns=["wav_file", "speaker"])
        df_empty.to_csv(MAPPING_FILE, index=False)
        show_status_message(
            "A new empty mapping file has been created.",
            style="green"
        )
        return df_empty
    else:
        try:
            return pd.read_csv(MAPPING_FILE)
        except Exception as e:
            show_status_message(
                f"Error: Failed to read {MAPPING_FILE}. {e}",
                style="red"
            )
            show_status_message(
                "Creating a new mapping file...",
                style="yellow"
            )
            df_empty = pd.DataFrame(columns=["wav_file", "speaker"])
            df_empty.to_csv(MAPPING_FILE, index=False)
            return df_empty

# File processing and counting
def get_files_to_process(df_mapping):
    """Get a list of JSON files that need processing."""
    # Get all JSON and WAV files
    json_files = set([f for f in os.listdir(JSON_DIR) if f.endswith('.json')])
    wav_files = set([f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')])
    
    # Extract base filenames for comparison
    json_bases = {os.path.splitext(f)[0] for f in json_files}
    wav_bases = {os.path.splitext(f)[0] for f in wav_files}
    
    # Find files that have both JSON and WAV
    common_bases = json_bases.intersection(wav_bases)
    
    # Get files that are not already in mappings
    processed_wav_files = set(df_mapping['wav_file'].tolist())
    
    # Files to process are those that have both JSON and WAV but aren't in mappings
    files_to_process = []
    for base in common_bases:
        wav_file = f"{base}.wav"
        if wav_file not in processed_wav_files:
            files_to_process.append(f"{base}.json")
    
    # Report stats - only show individual counts if there's a mismatch
    if len(json_files) != len(wav_files) or len(json_bases) != len(wav_bases):
        show_status_message(
            "WARNING: Mismatched files detected",
            style="yellow"
        )
        show_status_message(
            f"Total JSON files: {len(json_files)}",
            style="red"
        )
        show_status_message(
            f"Total WAV files: {len(wav_files)}",
            style="red"
        )
    else:
        show_status_message(
            f"Found {len(common_bases)} matching pairs of JSON and WAV files",
            style="green"
        )
    
    show_status_message(
        f"Files already processed: {len(processed_wav_files)}",
        style="green"
    )
    show_status_message(
        f"Files remaining to process: {len(files_to_process)}",
        style="green"
    )
    
    return files_to_process

# Process a single file
# Modify the process_file function to exit early when a target speaker is identified

def process_file(json_file, df_mapping):
    """Processes a single JSON file and extracts segments of target speakers."""
    json_path = os.path.join(JSON_DIR, json_file)
    wav_filename = os.path.splitext(json_file)[0] + ".wav"
    wav_path = os.path.join(AUDIO_DIR, wav_filename)
    
    show_status_message(f"Processing: {json_file}", style="cyan")
    
    if not os.path.exists(json_path):
        show_status_message(f"Error: JSON file not found at {json_path}", style="red")
        return 0, df_mapping, False
    
    if not os.path.exists(wav_path):
        show_status_message(f"Error: WAV file not found at {wav_path}", style="red")
        return 0, df_mapping, False
    
    # Load JSON data
    try:
        with open(json_path, "r") as f:
            segments = json.load(f)
    except Exception as e:
        show_status_message(f"Error reading JSON file {json_file}: {e}", style="red")
        return 0, df_mapping, False
    
    # Load audio data
    try:
        sample_rate, audio_data = wavfile.read(wav_path)
    except Exception as e:
        show_status_message(f"Error reading WAV file {wav_filename}: {e}", style="red")
        return 0, df_mapping, False
    
    # Get unique speakers and count segments for each
    speaker_segments = {}
    for segment in segments:
        speaker = segment.get('speaker', 'unknown')
        speaker_segments.setdefault(speaker, []).append(segment)
    
    num_speakers = len(speaker_segments)
    show_status_message(f"Found {num_speakers} unique speakers in the JSON file.", style="cyan")
    
    # Track identified speakers to avoid duplication
    identified_speakers = set()
    segment_count = 0
    speakers_checked = 0
    
    # Process each speaker's segments
    for speaker, speaker_segs in speaker_segments.items():
        # Skip if this speaker has been processed
        if speaker in identified_speakers:
            continue
        
        # Find valid segments (within length constraints)
        valid_segments = []
        for seg in speaker_segs:
            start_time = seg.get("start", 0)
            end_time = seg.get("end", 0)
            duration = end_time - start_time
            
            if MIN_CLIP_LENGTH <= duration <= MAX_CLIP_LENGTH:
                valid_segments.append(seg)
        
        if not valid_segments:
            show_status_message(f"No valid segments found for speaker '{speaker}'.", style="yellow")
            speakers_checked += 1
            continue
        
        # Choose a segment in the middle for better representation
        segment = valid_segments[len(valid_segments) // 2]
        
        start_sample = int(segment["start"] * sample_rate)
        end_sample = int(segment["end"] * sample_rate)
        
        # Ensure valid indices
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)
        
        if start_sample >= end_sample:
            show_status_message(f"Invalid segment time range for speaker '{speaker}'.", style="yellow")
            speakers_checked += 1
            continue
        
        segment_audio = audio_data[start_sample:end_sample]
        
        # Track the segments we've already seen for this speaker
        seen_segments = set()
        seen_segments.add(segment["start"])  # Add the initial segment
        is_repeating = False  # Flag to track if we're repeating segments
        
        # Loop for user interaction with this segment
        while True:
            clear_console()
            print_title()
            print_status(wav_filename, speaker, num_speakers, speakers_checked, segment, is_repeating)
            print_menu()
            
            # Play the audio segment
            play_audio(segment_audio, sample_rate)
            
            # Get user's decision
            user_input = get_user_decision()
            
            if user_input == "y":
                # Check if this wav file already has an entry
                existing = df_mapping[df_mapping['wav_file'] == wav_filename]
                if len(existing) > 0:
                    # Update existing entry
                    df_mapping.loc[df_mapping['wav_file'] == wav_filename, 'speaker'] = speaker
                else:
                    # Add new entry
                    new_row = pd.DataFrame({'wav_file': [wav_filename], 'speaker': [speaker]})
                    df_mapping = pd.concat([df_mapping, new_row], ignore_index=True)
                
                # Save updated mapping
                df_mapping.to_csv(MAPPING_FILE, index=False)
                
                show_status_message(
                    f"Speaker '{speaker}' has been identified as the target for {wav_filename}!",
                    style="green"
                )
                
                # NEW: Additional confirmation message about stopping file processing
                show_status_message(
                    f"Target speaker identified! Skipping remaining speakers in this file.",
                    style="green"
                )
                
                # Pause for a moment to let the user see the messages
                import time
                time.sleep(2)
                
                # Return immediately after identifying a target speaker
                return 1, df_mapping, False  # 1 segment, don't exit processing loop
            
            elif user_input == "n":
                show_status_message(
                    f"Speaker '{speaker}' has been marked as NOT the targeted speaker.",
                    style="yellow"
                )
                identified_speakers.add(speaker)  # Skip this speaker in future iterations
                speakers_checked += 1
                break
            
            elif user_input == "a":
                show_status_message(
                    f"Replaying the current clip for speaker '{speaker}'.",
                    style="blue"
                )
                continue
            
            elif user_input == "u":
                # Find the next valid segment for this speaker
                current_index = speaker_segs.index(segment)
                next_segments = [s for s in speaker_segs[current_index+1:] 
                                if MIN_CLIP_LENGTH <= (s["end"] - s["start"]) <= MAX_CLIP_LENGTH]
                
                if next_segments:
                    # We have more segments to go through
                    segment = next_segments[0]
                    # Check if we've already seen this segment
                    if segment["start"] in seen_segments:
                        is_repeating = True
                        show_status_message(
                            f"Now repeating segments for speaker '{speaker}'.",
                            style="yellow"
                        )
                    else:
                        seen_segments.add(segment["start"])
                        is_repeating = False  # Reset in case we previously set it to True
                        
                    start_sample = int(segment["start"] * sample_rate)
                    end_sample = int(segment["end"] * sample_rate)
                    # Ensure valid indices
                    start_sample = max(0, start_sample)
                    end_sample = min(len(audio_data), end_sample)
                    segment_audio = audio_data[start_sample:end_sample]
                    show_status_message(
                        f"Moving to the next segment for speaker '{speaker}'.",
                        style="blue"
                    )
                else:
                    # If we've run out of segments, start over from the beginning
                    valid_segments = [s for s in speaker_segs 
                                     if MIN_CLIP_LENGTH <= (s["end"] - s["start"]) <= MAX_CLIP_LENGTH]
                    
                    if valid_segments:
                        segment = valid_segments[0]
                        is_repeating = True  # Mark that we're now repeating segments
                        show_status_message(
                            f"Restarting from the beginning for speaker '{speaker}'. All segments will be repeated.",
                            style="yellow"
                        )
                        
                        start_sample = int(segment["start"] * sample_rate)
                        end_sample = int(segment["end"] * sample_rate)
                        # Ensure valid indices
                        start_sample = max(0, start_sample)
                        end_sample = min(len(audio_data), end_sample)
                        segment_audio = audio_data[start_sample:end_sample]
                        
                        # Reset seen_segments to only include the first one we're showing again
                        seen_segments = {segment["start"]}
                    else:
                        show_status_message(
                            f"No valid segments for speaker '{speaker}'.",
                            style="yellow"
                        )
                        break
            
            elif user_input == "x":
                show_status_message(
                    "Stopping processing. You can continue later.",
                    style="red"
                )
                return segment_count, df_mapping, True  # Added flag to indicate user wants to exit
            
            else:
                show_status_message(
                    "Invalid input. Please try again.",
                    style="yellow"
                )
    
    # Check if all speakers were evaluated but none were identified as the target
    if speakers_checked >= num_speakers and segment_count == 0:
        show_status_message(
            f"No targeted speaker identified in {wav_filename} after checking all {num_speakers} speakers.",
            style="red"
        )
        
        # Ask the user if they want to reprocess or skip
        console.print(Panel(
            Align("[bold yellow]Would you like to reprocess this file or skip?[/bold yellow]", "center"),
            border_style="yellow",
            width=CONSOLE_WIDTH
        ))
        reprocess_prompt = console.input("\n(R for reprocess/S for skip): ").strip().lower()
        
        if reprocess_prompt == "r":
            show_status_message(
                f"Reprocessing {wav_filename}...",
                style="blue"
            )
            return process_file(json_file, df_mapping)
        else:
            show_status_message(
                f"Skipping {wav_filename}. No entry will be made in mappings.csv.",
                style="yellow"
            )
            return 0, df_mapping, False
    
    return segment_count, df_mapping, False  # Added flag (False = don't exit processing loop)

# The actual process of the script - main code.
def main():
    """Main function to run the speaker identification tool."""
    clear_console()
    print_title()
    
    # Load or create the mapping file
    df_mapping = load_or_create_mapping_file()
    show_status_message(f"Loaded speaker mapping for {len(df_mapping)} files.", style="green")
    
    # Get files that need to be processed
    files_to_process = get_files_to_process(df_mapping)
    
    if not files_to_process:
        show_status_message("No new files to process. All files have been processed.", style="yellow")
        return
    
    prompt_start()
    
    # Process each file
    total_segments = 0
    processed_files = 0
    total_files = len(files_to_process)
    
    for json_file in files_to_process:
        show_status_message(f"Processing file {processed_files + 1}/{total_files}: {json_file}", style="cyan")
        
        segments, df_mapping, exit_requested = process_file(json_file, df_mapping)
        total_segments += segments
        processed_files += 1
        
        # If user requested to exit (by pressing 'X'), break the loop
        if exit_requested:
            show_status_message("Exiting as requested. Progress saved.", style="red")
            break
        
        # Ask if user wants to continue after each file (only if not the last file)
        if processed_files < total_files:
            console.print(Panel(
                Align("[bold yellow]Continue to next file?[/bold yellow]", "center"),
                border_style="yellow",
                width=CONSOLE_WIDTH
            ))
            continue_prompt = console.input("\n(y/n): ").strip().lower()
            
            if continue_prompt != "y":
                show_status_message("Processing paused. You can continue later.", style="red")
                break
    
    # Final summary
    show_status_message("Processing complete!", style="green")
    show_status_message(f"Files processed: {processed_files}/{total_files}", style="green")
    show_status_message(f"Total speakers identified: {total_segments}", style="green")
    show_status_message(f"Total files in mapping: {len(df_mapping)}", style="green")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print(Panel(
            Align("[bold red]Program interrupted by user. Exiting...[/bold red]", "center"),
            border_style="red",
            width=CONSOLE_WIDTH
        ))
    except Exception as e:
        console.print(Panel(
            Align(f"[bold red]An unexpected error occurred: {e}[/bold red]", "center"),
            border_style="red",
            width=CONSOLE_WIDTH
        ))
    finally:
        # Make sure we clean up sounddevice if necessary
        try:
            sd.stop()
        except:
            pass