import os
import json
import pandas as pd
import numpy as np
from scipy.io import wavfile
from rich.console import Console
import sounddevice as sd

# Initialize Rich Console
console = Console()

# Common line separator length
separator = "=" * 45

# Clear the terminal screen
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

# Title Screen
def print_title():
    console.print(
        f"{separator}\n[bold cyan] SPEAKER IDENTIFICATION TOOL [/bold cyan]\n{separator}\n",
        justify="left"
    )
    console.print(
        "This tool interactively identifies a targeted\nspeaker in an audio file, using JSON metadata.",
        justify="left"
    )
    console.print(
        "[italic yellow]NOTE: For the best results, run your audio\nthrough UVR or a similar isolation project.\nIf you don't, groans, machines wirring,\nbackground music vocals among other things\nwill get marked as a valid speaker.[/italic yellow]\n",
        justify="left"
    )

# Menu Display
def print_menu():
    console.print(
        f"{separator}\n[bold cyan] MENU OPTIONS [/bold cyan]\n{separator}\n",
        justify="left"
    )
    menu_block = f"""[bold yellow]
 [bold magenta]Y[/bold magenta] - Confirm that this is the Targeted Speaker
 [bold magenta]N[/bold magenta] - Mark this as NOT the Targeted Speaker
 [bold magenta]A[/bold magenta] - Listen to the same clip again
 [bold magenta]U[/bold magenta] - Move to the Next Segment for the Same Speaker
 [bold magenta]X[/bold magenta] - Stop and Continue Processing Files Later
 [/bold yellow]"""
    console.print(menu_block, justify="left")

# Prompt User to Start
def prompt_start():
    start_prompt = console.input(
        f"[bold yellow]{separator}\nAre you ready to start identifying speakers? (y/n):\n{separator}[/bold yellow]\n"
    ).strip().lower()
    while start_prompt not in ["y", "n"]:
        start_prompt = console.input(
            "[bold yellow]Invalid input. Please enter 'y' or 'n': [/bold yellow]"
        ).strip().lower()
    if start_prompt == "n":
        console.print("[bold red]Exiting tool. Goodbye![/bold red]")
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
        console.print(f"[red]Error playing audio: {e}[/red]")
        # Brief pause to show the error
        import time
        time.sleep(2)

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
        console.print(
            f"[yellow]Mapping file not found. Creating a new one at {MAPPING_FILE}...[/yellow]"
        )
        df_empty = pd.DataFrame(columns=["wav_file", "speaker"])
        df_empty.to_csv(MAPPING_FILE, index=False)
        console.print(
            "[green]A new empty mapping file has been created.[/green]"
        )
        return df_empty
    else:
        try:
            return pd.read_csv(MAPPING_FILE)
        except Exception as e:
            console.print(f"[red]Error: Failed to read {MAPPING_FILE}. {e}[/red]")
            console.print("[yellow]Creating a new mapping file...[/yellow]")
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
        console.print(f"[yellow]WARNING: Mismatched files detected[/yellow]")
        console.print(f"[red]Total JSON files: {len(json_files)}[/red]")
        console.print(f"[red]Total WAV files: {len(wav_files)}[/red]")
    else:
        console.print(f"[green]Found {len(common_bases)} matching pairs of JSON and WAV files[/green]")
    
    console.print(f"[bold green]Files already processed: {len(processed_wav_files)}[/bold green]")
    console.print(f"[bold green]Files remaining to process: {len(files_to_process)}[/bold green]")
    
    return files_to_process

# Memory load mappings and numbers from above, faster processing
def process_file(json_file, df_mapping):
    """Processes a single JSON file and extracts segments of target speakers."""
    json_path = os.path.join(JSON_DIR, json_file)
    wav_filename = os.path.splitext(json_file)[0] + ".wav"
    wav_path = os.path.join(AUDIO_DIR, wav_filename)
    
    console.print(f"[bold cyan]Processing: {json_file}[/bold cyan]")
    
    if not os.path.exists(json_path):
        console.print(f"[red]Error: JSON file not found at {json_path}[/red]")
        return 0, df_mapping, False
    
    if not os.path.exists(wav_path):
        console.print(f"[red]Error: WAV file not found at {wav_path}[/red]")
        return 0, df_mapping, False
    
    # Load JSON data
    try:
        with open(json_path, "r") as f:
            segments = json.load(f)
    except Exception as e:
        console.print(f"[red]Error reading JSON file {json_file}: {e}[/red]")
        return 0, df_mapping, False
    
    # Load audio data
    try:
        sample_rate, audio_data = wavfile.read(wav_path)
    except Exception as e:
        console.print(f"[red]Error reading WAV file {wav_filename}: {e}[/red]")
        return 0, df_mapping, False
    
    # Get unique speakers and count segments for each
    speaker_segments = {}
    for segment in segments:
        speaker = segment.get('speaker', 'unknown')
        speaker_segments.setdefault(speaker, []).append(segment)
    
    num_speakers = len(speaker_segments)
    console.print(f"[cyan]Found {num_speakers} unique speakers in the JSON file.[/cyan]")
    
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
            console.print(f"[yellow]No valid segments found for speaker '{speaker}'.[/yellow]")
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
            console.print(f"[yellow]Invalid segment time range for speaker '{speaker}'.[/yellow]")
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
            console.print(f"[cyan]Current WAV File: {wav_filename}[/cyan]")
            
            # Add warning if we're repeating segments
            if is_repeating:
                console.print(f"[cyan]Current Speaker: {speaker} ({len(speaker_segs)} segments) [bold red](REPEATING SEGMENTS)[/bold red][/cyan]")
            else:
                console.print(f"[cyan]Current Speaker: {speaker} ({len(speaker_segs)} segments)[/cyan]")
                
            console.print(f"[cyan]Number of Speakers in JSON: {num_speakers}[/cyan]")
            console.print(f"[cyan]Speakers Checked: {speakers_checked}/{num_speakers}[/cyan]")
            console.print(f"[cyan]Segment Time: {segment['start']:.2f}s - {segment['end']:.2f}s[/cyan]")
            print_menu()
            
            # Play the audio segment
            play_audio(segment_audio, sample_rate)
            
            # Get user's decision
            user_input = console.input(
                f"[bold yellow]{separator}\nIs this the targeted speaker? (Y/N/A/U/X):\n{separator}[/bold yellow]\n"
            ).strip().lower()
            
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
                
                console.print(f"[green]Speaker '{speaker}' has been identified as the target for {wav_filename}![/green]")
                identified_speakers.add(speaker)
                segment_count += 1
                speakers_checked += 1
                break
            
            elif user_input == "n":
                console.print(f"[yellow]Speaker '{speaker}' has been marked as NOT the targeted speaker.[/yellow]")
                identified_speakers.add(speaker)  # Skip this speaker in future iterations
                speakers_checked += 1
                break
            
            elif user_input == "a":
                console.print(f"[blue]Replaying the current clip for speaker '{speaker}'.[/blue]")
                continue
            
            elif user_input == "u":
                # Find the next valid segment for this speaker
                current_index = speaker_segs.index(segment)
                next_segments = [s for s in speaker_segs[current_index+1:] 
                                if MIN_CLIP_LENGTH <= (s["end"] - s["start"]) <= MAX_CLIP_LENGTH]
                
                if next_segments:
                    segment = next_segments[0]
                    # Check if we've already seen this segment
                    if segment["start"] in seen_segments:
                        is_repeating = True
                    else:
                        seen_segments.add(segment["start"])
                        
                    start_sample = int(segment["start"] * sample_rate)
                    end_sample = int(segment["end"] * sample_rate)
                    # Ensure valid indices
                    start_sample = max(0, start_sample)
                    end_sample = min(len(audio_data), end_sample)
                    segment_audio = audio_data[start_sample:end_sample]
                    console.print(f"[blue]Moving to the next segment for speaker '{speaker}'.[/blue]")
                else:
                    # If we've run out of segments, start over from the beginning
                    valid_segments = [s for s in speaker_segs 
                                     if MIN_CLIP_LENGTH <= (s["end"] - s["start"]) <= MAX_CLIP_LENGTH]
                    
                    if valid_segments:
                        segment = valid_segments[0]
                        is_repeating = True  # Mark that we're now repeating segments
                        
                        start_sample = int(segment["start"] * sample_rate)
                        end_sample = int(segment["end"] * sample_rate)
                        # Ensure valid indices
                        start_sample = max(0, start_sample)
                        end_sample = min(len(audio_data), end_sample)
                        segment_audio = audio_data[start_sample:end_sample]
                        console.print(f"[yellow]Restarting from the beginning for speaker '{speaker}'.[/yellow]")
                    else:
                        console.print(f"[yellow]No valid segments for speaker '{speaker}'.[/yellow]")
                        break
            
            elif user_input == "x":
                console.print("[red]Stopping processing. You can continue later.[/red]")
                return segment_count, df_mapping, True  # Added flag to indicate user wants to exit
            
            else:
                console.print("[yellow]Invalid input. Please try again.[/yellow]")
    
    # Check if all speakers were evaluated but none were identified as the target
    if speakers_checked >= num_speakers and segment_count == 0:
        console.print(f"[bold red]No targeted speaker identified in {wav_filename} after checking all {num_speakers} speakers.[/bold red]")
        
        # Ask the user if they want to reprocess or skip
        reprocess_prompt = console.input(
            f"[bold yellow]{separator}\nWould you like to reprocess this file or skip? (R for reprocess/S for skip):\n{separator}[/bold yellow]\n"
        ).strip().lower()
        
        if reprocess_prompt == "r":
            console.print(f"[blue]Reprocessing {wav_filename}...[/blue]")
            return process_file(json_file, df_mapping)
        else:
            console.print(f"[yellow]Skipping {wav_filename}. No entry will be made in mappings.csv.[/yellow]")
            return 0, df_mapping, False
    
    return segment_count, df_mapping, False  # Added flag (False = don't exit processing loop)

# The actual process of the script - main code.
def main():
    """Main function to run the speaker identification tool."""
    clear_console()
    print_title()
    
    # Load or create the mapping file
    df_mapping = load_or_create_mapping_file()
    console.print(f"[bold green]Loaded speaker mapping for {len(df_mapping)} files.[/bold green]\n")
    
    # Get files that need to be processed
    files_to_process = get_files_to_process(df_mapping)
    
    if not files_to_process:
        console.print("[yellow]No new files to process. All files have been processed.[/yellow]")
        return
    
    prompt_start()
    
    # Process each file
    total_segments = 0
    processed_files = 0
    total_files = len(files_to_process)
    
    for json_file in files_to_process:
        console.print(f"[bold cyan]Processing file {processed_files + 1}/{total_files}: {json_file}[/bold cyan]")
        
        segments, df_mapping, exit_requested = process_file(json_file, df_mapping)
        total_segments += segments
        processed_files += 1
        
        # If user requested to exit (by pressing 'X'), break the loop
        if exit_requested:
            console.print("[red]Exiting as requested. Progress saved.[/red]")
            break
        
        # Ask if user wants to continue after each file (only if not the last file)
        if processed_files < total_files:
            continue_prompt = console.input(
                f"[bold yellow]{separator}\nContinue to next file? (y/n):\n{separator}[/bold yellow]\n"
            ).strip().lower()
            
            if continue_prompt != "y":
                console.print("[red]Processing paused. You can continue later.[/red]")
                break
    
    # Final summary
    console.print(f"[bold green]Processing complete![/bold green]")
    console.print(f"[green]Files processed: {processed_files}/{total_files}[/green]")
    console.print(f"[green]Total speakers identified: {total_segments}[/green]")
    console.print(f"[green]Total files in mapping: {len(df_mapping)}[/green]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Program interrupted by user. Exiting...[/bold red]")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
    finally:
        # Make sure we clean up sounddevice if necessary
        try:
            sd.stop()
        except:
            pass