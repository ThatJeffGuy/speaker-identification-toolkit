import os
import json
import pandas as pd
from rich.console import Console
from pydub import AudioSegment

# Initialize Rich Console
console = Console()
console.clear()

separator = "=" * 45

# Title Screen
console.print(
    f"[bold cyan]\n{separator}\n       SPEAKER IDENTIFICATION TOOL\n{separator}[/bold cyan]\n",
    justify="center"
)
console.print(
    "This tool identifies targeted and not-targeted speakers in audio files based on JSON metadata.",
    justify="center"
)
console.print(
    "[italic yellow]Segments you've already processed won't be replayed.[/italic yellow]\n",
    justify="center"
)

# Menu Display
console.print(
    f"[bold cyan]\n{separator}\n             MENU OPTIONS\n{separator}[/bold cyan]\n",
    justify="center"
)

menu_block = f"""[bold yellow]
[bold magenta]Y[/bold magenta] - Confirm this Speaker is Targeted (Save and move on to next episode)
[bold magenta]N[/bold magenta] - Not the Targeted Speaker (Skip all future segments of this speaker, move to next speaker)
[bold magenta]T[/bold magenta] - Next Segment by the Same Speaker (If none, treat like N)
[bold magenta]L[/bold magenta] - Replay the Current Clip
[bold magenta]X[/bold magenta] - Exit Without Saving
[/bold yellow]"""

console.print(menu_block, justify="center")

start_prompt = console.input(
    f"[bold yellow]{separator}\nAre you ready to start identifying speakers? (y/n):\n{separator}[/bold yellow]\n"
).strip().lower()

while start_prompt not in ["y", "n"]:
    start_prompt = console.input("[bold yellow]Invalid input. Please enter 'y' or 'n': [/bold yellow]").strip().lower()

if start_prompt == "n":
    console.print("[bold red]Exiting tool. Goodbye![/bold red]", justify="center")
    exit(0)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(SCRIPT_DIR, "jsons")
AUDIO_DIR = os.path.join(SCRIPT_DIR, "wavs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "targeted")
MAPPING_FILE = os.path.join(SCRIPT_DIR, "mappings.csv")

os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(MAPPING_FILE):
    console.print(
        f"[yellow]Mapping file not found. Creating a new one at {MAPPING_FILE}...[/yellow]",
        justify="center"
    )
    df_empty = pd.DataFrame(columns=["json_file", "target_speaker_label"])
    df_empty.to_csv(MAPPING_FILE, index=False)
    console.print(
        "[green]A new empty mapping file has been created.[/green]",
        justify="center"
    )

try:
    df_mapping = pd.read_csv(MAPPING_FILE)
except Exception as e:
    console.print(f"[red]Error: Failed to read {MAPPING_FILE}. {e}[/red]", justify="center")
    df_mapping = pd.DataFrame(columns=["json_file", "target_speaker_label"])

console.print(
    f"[bold green]Loaded speaker mapping for {len(df_mapping)} files.[/bold green]\n",
    justify="center"
)

def check_ffplay():
    try:
        os.system("ffplay -version > /dev/null 2>&1")
    except OSError:
        console.print("[red]Error: ffplay not found. Ensure it is installed and in your PATH.[/red]", justify="center")
        exit(1)

check_ffplay()

def save_mapping(json_file, speaker_label):
    global df_mapping
    new_row = {"json_file": json_file, "target_speaker_label": speaker_label}
    df_mapping = pd.concat([df_mapping, pd.DataFrame([new_row])], ignore_index=True)
    df_mapping.to_csv(MAPPING_FILE, index=False)
    console.print(f"[green]Saved mapping: {json_file} -> {speaker_label}[/green]", justify="center")

def play_clip(temp_file):
    return os.system(f"ffplay -nodisp -autoexit -v quiet -hide_banner {temp_file}")

def replay_clip(temp_file):
    console.print("[cyan]Replaying the current clip...[/cyan]", justify="center")
    play_clip(temp_file)

def load_segments(json_path):
    with open(json_path, "r") as f:
        segments = json.load(f)
    # Initialize identified_as field if not present
    for seg in segments:
        if "identified_as" not in seg:
            seg["identified_as"] = "unprocessed"
    return segments

def save_segments(json_path, segments):
    with open(json_path, "w") as f:
        json.dump(segments, f, indent=2)

def remove_future_segments_of_speaker(segments, current_index, speaker_to_remove):
    # Mark all future segments of this speaker as not_targeted
    for i in range(current_index+1, len(segments)):
        if segments[i]["speaker"] == speaker_to_remove and segments[i]["identified_as"] == "unprocessed":
            segments[i]["identified_as"] = "not_targeted"
    return segments

def next_segment_same_speaker(segments, current_index, speaker):
    for i in range(current_index + 1, len(segments)):
        if segments[i]["speaker"] == speaker and segments[i]["identified_as"] == "unprocessed":
            return i
    return None

def next_unprocessed_segment(segments, from_index):
    # Move forward and find the next unprocessed segment
    for i in range(from_index+1, len(segments)):
        if segments[i]["identified_as"] == "unprocessed":
            return i
    return None

def process_file(json_file):
    json_path = os.path.join(JSON_DIR, json_file)
    wav_file = os.path.join(AUDIO_DIR, os.path.splitext(json_file)[0] + ".wav")

    if not os.path.exists(json_path) or not os.path.exists(wav_file):
        console.print(
            f"[yellow]Warning: Missing JSON or WAV file for {json_file}. Skipping...[/yellow]",
            justify="center"
        )
        return

    segments = load_segments(json_path)

    if not segments:
        console.print(f"[yellow]No segments found in {json_file}. Skipping...[/yellow]", justify="center")
        return

    audio = AudioSegment.from_wav(wav_file)

    # Start from the first unprocessed segment
    segment_index = 0
    # Find the first unprocessed segment
    while segment_index < len(segments) and segments[segment_index]["identified_as"] != "unprocessed":
        segment_index += 1

    while segment_index < len(segments):
        segment = segments[segment_index]

        # If processed, move to next unprocessed
        if segment["identified_as"] != "unprocessed":
            segment_index = next_unprocessed_segment(segments, segment_index)
            if segment_index is None:
                # No more unprocessed segments, this episode done
                break
            continue

        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        
        # Skip very short segments
        if (end_ms - start_ms) < 1000:
            segments[segment_index]["identified_as"] = "not_targeted"  # too short to be useful, mark as not_targeted
            save_segments(json_path, segments)
            segment_index = next_unprocessed_segment(segments, segment_index)
            if segment_index is None:
                break
            continue

        current_speaker = segment["speaker"]

        # Export and play current segment
        temp_file = os.path.join(SCRIPT_DIR, "temp_clip.wav")
        segment_audio = audio[start_ms:end_ms]
        segment_audio.export(temp_file, format="wav")

        console.print(
            f"[cyan]Playing segment {start_ms / 1000:.2f}-{end_ms / 1000:.2f}s | Speaker: {current_speaker}[/cyan]",
            justify="center"
        )
        play_status = play_clip(temp_file)
        if play_status != 0:
            console.print("[red]Warning: ffplay encountered an issue playing this segment.[/red]", justify="center")

        # Get user choice
        while True:
            choice = console.input("[bold yellow]Enter your choice (Y/N/T/L/X): [/bold yellow]").strip().lower()

            if choice == "y":
                # This speaker is targeted for the entire episode
                segments[segment_index]["identified_as"] = "targeted"
                save_segments(json_path, segments)
                save_mapping(json_file, current_speaker)
                return  # done with this episode

            elif choice == "n":
                # Not targeted: mark this segment as not_targeted
                segments[segment_index]["identified_as"] = "not_targeted"
                # Remove future segments of this speaker
                segments = remove_future_segments_of_speaker(segments, segment_index, current_speaker)
                save_segments(json_path, segments)
                # Move to next unprocessed segment of a different speaker
                next_index = next_unprocessed_segment(segments, segment_index)
                if next_index is None:
                    return  # no more segments in this episode
                segment_index = next_index
                break

            elif choice == "t":
                # Next segment of the same speaker
                next_same = next_segment_same_speaker(segments, segment_index, current_speaker)
                if next_same is not None:
                    # Move directly to that segment
                    segment_index = next_same
                else:
                    # Treat like N
                    segments[segment_index]["identified_as"] = "not_targeted"
                    segments = remove_future_segments_of_speaker(segments, segment_index, current_speaker)
                    save_segments(json_path, segments)
                    next_index = next_unprocessed_segment(segments, segment_index)
                    if next_index is None:
                        return  # no more segments in this episode
                    segment_index = next_index
                break

            elif choice == "l":
                # Replay the segment
                replay_clip(temp_file)

            elif choice == "x":
                # Exit without saving
                console.print("[bold red]Exiting without saving.[/bold red]", justify="center")
                exit(0)

            else:
                console.print("[red]Invalid choice. Please try again.[/red]", justify="center")

    console.print(f"[green]Finished processing {json_file}. Moving on...[/green]", justify="center")


console.print("[bold cyan]Starting processing...[/bold cyan]\n", justify="center")

for json_file in sorted(os.listdir(JSON_DIR)):
    if json_file.endswith(".json"):
        process_file(json_file)

console.print(
    f"[bold green]{separator}\nProcessing complete!\nCheck the updated mappings in: {MAPPING_FILE}\n{separator}\n",
    justify="center"
)

