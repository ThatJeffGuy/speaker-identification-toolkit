import os
import json
import numpy as np
import pandas as pd
from pydub import AudioSegment
from scipy.io import wavfile
from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.table import Table
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TextColumn
import torch
from sklearn.cluster import AgglomerativeClustering
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import pickle
import sounddevice as sd
import time
from pathlib import Path

# Improved CUDA detection and setup
def setup_cuda():
    """Setup CUDA environment with proper error handling and device selection."""
    if torch.cuda.is_available():
        # Get available CUDA devices
        num_devices = torch.cuda.device_count()
        device_names = [torch.cuda.get_device_name(i) for i in range(num_devices)]
        
        console.print(f"[bold green]CUDA is available with {num_devices} device(s):[/bold green]")
        for i, name in enumerate(device_names):
            console.print(f"[green]  - Device {i}: {name}[/green]")
        
        # Use the first CUDA device by default
        device = torch.device("cuda:0")
        
        # Set memory management for better performance
        torch.cuda.empty_cache()
        
        # Return device and info
        return device, {
            "available": True,
            "device_count": num_devices,
            "devices": device_names,
            "selected": 0
        }
    else:
        console.print("[yellow]CUDA is not available. Using CPU for processing (slower).[/yellow]")
        return torch.device("cpu"), {
            "available": False
        }

# Initialize Rich Console with a fixed width
CONSOLE_WIDTH = 60
console = Console(width=CONSOLE_WIDTH)

# Constants
SEPARATOR = "=" * CONSOLE_WIDTH

# Clear the terminal screen
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

# Title Screen
def print_title():
    console.print(Panel(
        Align("[bold cyan]CROSS-FILE SPEAKER RECOGNITION TOOL[/bold cyan]", "center"),
        border_style="cyan",
        width=CONSOLE_WIDTH
    ))
    
    note_text = (
        "This tool identifies speakers across multiple audio files "
        "by creating voice embeddings and clustering them."
    )
    
    console.print(Panel(
        Align(note_text, "center"),
        border_style="blue",
        width=CONSOLE_WIDTH
    ))

# Define Paths Relative to Script Location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(SCRIPT_DIR, "jsons")
AUDIO_DIR = os.path.join(SCRIPT_DIR, "wavs")
MAPPING_FILE = os.path.join(SCRIPT_DIR, "mappings.csv")
EMBEDDINGS_DIR = os.path.join(SCRIPT_DIR, "embeddings")
GLOBAL_MAPPING_FILE = os.path.join(SCRIPT_DIR, "global_mappings.csv")
MODEL_CACHE_DIR = os.path.join(SCRIPT_DIR, ".model_cache")

# Ensure directories exist
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Retry decorator for model loading
def retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise
                    console.print(f"[yellow]Retrying {func.__name__}. Error: {e}[/yellow]")
                    sleep_time = backoff_in_seconds * 2 ** x
                    time.sleep(sleep_time)
                    x += 1
        return wrapper
    return decorator

# Load speaker embedding model
@retry_with_backoff(retries=3)
def load_embedding_model(device):
    """Load speaker embedding model with proper CUDA support."""
    console.print("[cyan]Loading speaker embedding model...[/cyan]")
    
    # Track memory usage before loading model
    if device.type == "cuda":
        before_mem = torch.cuda.memory_allocated(device)
        console.print(f"[cyan]CUDA memory before loading model: {before_mem / 1024**2:.2f} MB[/cyan]")
    
    try:
        # First try to import SpeechBrain which is already in your requirements
        from speechbrain.pretrained import EncoderClassifier
        
        # Use SpeechBrain's speaker embedding model
        # Cache the model locally to avoid repeated downloads
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir=MODEL_CACHE_DIR,
            run_opts={"device": device}
        )
        
        # Track memory usage after loading model
        if device.type == "cuda":
            after_mem = torch.cuda.memory_allocated(device)
            console.print(f"[cyan]CUDA memory after loading model: {after_mem / 1024**2:.2f} MB[/cyan]")
            console.print(f"[cyan]Model memory usage: {(after_mem - before_mem) / 1024**2:.2f} MB[/cyan]")
        
        console.print("[green]Successfully loaded SpeechBrain speaker embedding model.[/green]")
        return model, "speechbrain"
    except Exception as sb_error:
        console.print(f"[yellow]Failed to load SpeechBrain model: {sb_error}[/yellow]")
        
        try:
            # Fallback to pyannote which you already use in diarize-dataset.py
            from pyannote.audio import Pipeline
            
            # Try to load token
            token_file = os.path.join(SCRIPT_DIR, ".hf_token")
            if os.path.exists(token_file):
                with open(token_file, "r") as f:
                    token = f.read().strip()
            else:
                console.print("[bold yellow]Hugging Face token required.[/bold yellow]")
                token = input("Please enter your Hugging Face token: ").strip()
                with open(token_file, "w") as f:
                    f.write(token)
            
            # Use pyannote's speaker embedding model
            model = Pipeline.from_pretrained("pyannote/embedding", use_auth_token=token)
            model.to(device)
            
            # Track memory usage after loading model
            if device.type == "cuda":
                after_mem = torch.cuda.memory_allocated(device)
                console.print(f"[cyan]CUDA memory after loading model: {after_mem / 1024**2:.2f} MB[/cyan]")
                console.print(f"[cyan]Model memory usage: {(after_mem - before_mem) / 1024**2:.2f} MB[/cyan]")
            
            console.print(f"[green]Successfully loaded pyannote speaker embedding model on {device}.[/green]")
            return model, "pyannote"
        
        except Exception as pa_error:
            console.print(f"[red]Failed to load pyannote model: {pa_error}[/red]")
            raise Exception("Failed to load any embedding model.")

# Batch processing for better CUDA efficiency
def process_batch(audio_batch, model, model_type, device, batch_size=8):
    """Process a batch of audio segments to extract embeddings more efficiently."""
    if model_type == "speechbrain":
        # Create a batch tensor
        max_length = max(len(audio) for audio in audio_batch)
        batch_tensor = torch.zeros(len(audio_batch), max_length).to(device)
        
        for i, audio in enumerate(audio_batch):
            waveform = torch.tensor(audio).float()
            if waveform.max() > 1.0:
                waveform = waveform / 32768.0  # for 16-bit audio
            batch_tensor[i, :len(audio)] = waveform
        
        # Process in smaller batches if needed
        embeddings = []
        for i in range(0, len(audio_batch), batch_size):
            mini_batch = batch_tensor[i:i+batch_size]
            with torch.no_grad():
                batch_emb = model.encode_batch(mini_batch)
                embeddings.extend(batch_emb.cpu().numpy())
        
        return embeddings
    
    elif model_type == "pyannote":
        # pyannote doesn't support true batching as simply, so we process sequentially
        # but still use the GPU efficiently
        embeddings = []
        
        # Create temporary files for each sample
        temp_files = []
        for i, audio in enumerate(audio_batch):
            # Convert to float32 and normalize
            audio_float = audio.astype(np.float32)
            if np.max(np.abs(audio_float)) > 1.0:
                audio_float = audio_float / 32768.0
            
            temp_file = os.path.join(EMBEDDINGS_DIR, f"temp_segment_{i}.wav")
            wavfile.write(temp_file, 44100, audio_float)  # Assuming 44.1kHz
            temp_files.append(temp_file)
        
        # Process files
        with torch.no_grad():
            for temp_file in temp_files:
                embedding = model({"audio": temp_file})
                embeddings.append(embedding.squeeze().cpu().numpy())
        
        # Clean up temp files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return embeddings
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Extract embedding using appropriate model
def extract_embedding(audio_data, sample_rate, model, model_type, device):
    if model_type == "speechbrain":
        # SpeechBrain expects a waveform tensor
        waveform = torch.tensor(audio_data).float().to(device)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Normalize if needed
        if waveform.max() > 1.0:
            waveform = waveform / 32768.0  # for 16-bit audio
        
        with torch.no_grad():
            embeddings = model.encode_batch(waveform)
            return embeddings.squeeze().cpu().numpy()
    
    elif model_type == "pyannote":
        # pyannote expects a path or a waveform
        # Convert to float32 and normalize
        audio_float = audio_data.astype(np.float32)
        if np.max(np.abs(audio_float)) > 1.0:
            audio_float = audio_float / 32768.0  # normalize for 16-bit audio
        
        # Create a temporary file
        temp_file = os.path.join(EMBEDDINGS_DIR, "temp_segment.wav")
        wavfile.write(temp_file, sample_rate, audio_float)
        
        with torch.no_grad():
            embedding = model({"audio": temp_file})
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return embedding.squeeze().cpu().numpy()
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Function to extract embeddings for a given file and its identified speaker
def process_file_embeddings(file_data, model, model_type, device, batch_size=8):
    """Process a single file to extract embeddings for the identified speaker with batch processing."""
    wav_file, target_speaker = file_data
    json_file = os.path.splitext(wav_file)[0] + ".json"
    
    json_path = os.path.join(JSON_DIR, json_file)
    wav_path = os.path.join(AUDIO_DIR, wav_file)
    output_path = os.path.join(EMBEDDINGS_DIR, f"{os.path.splitext(wav_file)[0]}_embeddings.pkl")
    
    # Skip if embeddings already exist
    if os.path.exists(output_path):
        return wav_file, 1, f"Embeddings already exist"
    
    # Check if files exist
    if not os.path.exists(json_path) or not os.path.exists(wav_path):
        return wav_file, 0, f"Missing JSON or WAV file"
    
    try:
        # Load JSON data
        with open(json_path, "r") as f:
            segments = json.load(f)
        
        # Load audio file
        sample_rate, audio_data = wavfile.read(wav_path)
        
        # Filter segments for target speaker
        target_segments = [seg for seg in segments if seg.get('speaker') == target_speaker]
        
        if not target_segments:
            return wav_file, 0, f"No segments for target speaker '{target_speaker}'"
        
        # Prepare batches for processing
        audio_segments = []
        segment_metadata = []
        
        for segment in target_segments:
            start_sample = int(segment["start"] * sample_rate)
            end_sample = int(segment["end"] * sample_rate)
            
            # Skip very short segments
            if end_sample - start_sample < sample_rate:  # less than 1 second
                continue
                
            # Extract audio segment
            segment_audio = audio_data[start_sample:end_sample]
            
            audio_segments.append(segment_audio)
            segment_metadata.append({
                "file": wav_file,
                "speaker": target_speaker,
                "start": segment["start"],
                "end": segment["end"]
            })
        
        # Process in batches for better GPU utilization
        embeddings = []
        for i in range(0, len(audio_segments), batch_size):
            batch_audio = audio_segments[i:i+batch_size]
            batch_metadata = segment_metadata[i:i+batch_size]
            
            try:
                # Process batch
                if len(batch_audio) == 1:
                    # Single item, use regular processing
                    embedding = extract_embedding(batch_audio[0], sample_rate, model, model_type, device)
                    embeddings.append({
                        **batch_metadata[0],
                        "embedding": embedding
                    })
                else:
                    # Process as a batch
                    batch_embeddings = process_batch(batch_audio, model, model_type, device, batch_size)
                    for j, embedding in enumerate(batch_embeddings):
                        if j < len(batch_metadata):  # Safety check
                            embeddings.append({
                                **batch_metadata[j],
                                "embedding": embedding
                            })
            except Exception as e:
                console.print(f"[yellow]Error processing batch in {wav_file}: {e}[/yellow]")
        
        # Save embeddings
        with open(output_path, "wb") as f:
            pickle.dump(embeddings, f)
        
        return wav_file, len(embeddings), "Success"
    
    except Exception as e:
        return wav_file, 0, f"Error: {e}"

# Function to cluster embeddings across files
def cluster_embeddings(embeddings_list, n_clusters=None):
    """Cluster embeddings across files to identify the same speakers."""
    if not embeddings_list:
        return []
    
    # Extract embeddings and metadata
    X = []
    metadata = []
    
    for file_emb in embeddings_list:
        for emb in file_emb:
            X.append(emb["embedding"])
            metadata.append({
                "file": emb["file"],
                "original_speaker": emb["speaker"],
                "start": emb["start"],
                "end": emb["end"]
            })
    
    # Convert to numpy array
    X = np.array(X)
    
    # Determine number of clusters if not specified
    if n_clusters is None:
        # Estimate based on number of unique original speaker labels
        unique_original_speakers = len(set(m["original_speaker"] for m in metadata))
        
        # Set a reasonable range - between half and double the original count
        min_clusters = max(2, unique_original_speakers // 2)
        max_clusters = min(unique_original_speakers * 2, len(X) // 2)
        
        # Use a heuristic or let the user decide
        n_clusters = min(max(min_clusters, 
                           int(unique_original_speakers * 1.2)), 
                       max_clusters)
        
        console.print(f"[yellow]Estimated number of global speakers: {n_clusters}[/yellow]")
        
        # Let user override if they want
        user_input = console.input(f"[bold yellow]Enter custom number of speakers (or press Enter to use {n_clusters}): [/bold yellow]")
        if user_input.strip() and user_input.strip().isdigit():
            n_clusters = int(user_input.strip())
            console.print(f"[green]Using {n_clusters} clusters as specified.[/green]")
    
    # Cluster the embeddings
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity='cosine',
        linkage='average'
    )
    
    # Perform clustering
    labels = clustering.fit_predict(X)
    
    # Add cluster labels to metadata
    for i, label in enumerate(labels):
        metadata[i]["global_speaker"] = f"Speaker_{label+1}"
    
    return metadata

# Function to play audio clip for verification
def play_audio_clip(wav_file, start_time, end_time):
    """Play a specific segment of an audio file."""
    try:
        wav_path = os.path.join(AUDIO_DIR, wav_file)
        
        # Load the audio file
        sample_rate, audio_data = wavfile.read(wav_path)
        
        # Calculate start and end samples
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Extract segment
        segment_audio = audio_data[start_sample:end_sample]
        
        # Convert to float32 if needed
        if segment_audio.dtype != np.float32:
            segment_audio = segment_audio.astype(np.float32)
            # Normalize if int type was converted to float
            if np.max(np.abs(segment_audio)) > 1.0:
                segment_audio = segment_audio / 32768.0  # normalize for 16-bit audio
        
        # Play the audio
        sd.play(segment_audio, sample_rate)
        sd.wait()
        
        return True
    except Exception as e:
        console.print(f"[red]Error playing audio: {e}[/red]")
        return False

# Function to let user verify clusters
def verify_clusters(cluster_metadata):
    """Let user verify and adjust cluster assignments."""
    # Group by cluster
    clusters = {}
    for item in cluster_metadata:
        speaker = item["global_speaker"]
        clusters.setdefault(speaker, []).append(item)
    
    # Create cluster examples (sample a few segments from each cluster)
    cluster_examples = {}
    for speaker, items in clusters.items():
        # Take up to 3 items from different files if possible
        files = set()
        examples = []
        
        for item in items:
            if item["file"] not in files and len(examples) < 3:
                examples.append(item)
                files.add(item["file"])
        
        # If we didn't get 3 different files, just take the first 3 items
        if len(examples) < 3:
            examples = items[:3]
            
        cluster_examples[speaker] = examples
    
    # Let user verify each cluster
    verified_clusters = {}
    
    for speaker, examples in cluster_examples.items():
        clear_console()
        console.print(Panel(
            Align(f"[bold cyan]Verifying Cluster: {speaker}[/bold cyan]", "center"),
            border_style="cyan",
            width=CONSOLE_WIDTH
        ))
        
        console.print(f"[yellow]Playing {len(examples)} sample segments from this cluster...[/yellow]")
        
        # Show examples table
        table = Table(title=f"Sample Segments for {speaker}")
        table.add_column("Sample", style="cyan")
        table.add_column("File", style="green")
        table.add_column("Original Speaker", style="yellow")
        table.add_column("Time Range", style="magenta")
        
        for i, example in enumerate(examples, 1):
            table.add_row(
                str(i),
                example["file"],
                example["original_speaker"],
                f"{example['start']:.2f}s - {example['end']:.2f}s"
            )
        
        console.print(table)
        
        # Play each example
        for i, example in enumerate(examples, 1):
            console.print(f"[bold cyan]Playing sample {i}...[/bold cyan]")
            play_audio_clip(example["file"], example["start"], example["end"])
            time.sleep(0.5)  # small pause between clips
        
        # Ask user for confirmation
        user_input = console.input(f"\n[bold yellow]Is this a consistent voice throughout all samples? (y/n): [/bold yellow]").strip().lower()
        
        if user_input == 'y':
            # Ask for a name for this speaker
            speaker_name = console.input(f"[bold yellow]Enter a name for this speaker (leave empty to use {speaker}): [/bold yellow]").strip()
            if not speaker_name:
                speaker_name = speaker
            
            verified_clusters[speaker] = {
                "new_name": speaker_name,
                "is_verified": True
            }
        else:
            # If user says it's not consistent, mark for manual review
            verified_clusters[speaker] = {
                "new_name": f"REVIEW_{speaker}",
                "is_verified": False
            }
    
    # Apply verified labels to metadata
    for item in cluster_metadata:
        speaker = item["global_speaker"]
        if speaker in verified_clusters:
            item["global_speaker"] = verified_clusters[speaker]["new_name"]
            item["is_verified"] = verified_clusters[speaker]["is_verified"]
    
    return cluster_metadata, verified_clusters

# Save Global Speaker Mapping
def save_global_mapping(cluster_metadata):
    """Save global speaker mapping to CSV file."""
    # Create a DataFrame from cluster metadata
    df = pd.DataFrame({
        "file": [item["file"] for item in cluster_metadata],
        "original_speaker": [item["original_speaker"] for item in cluster_metadata],
        "global_speaker": [item["global_speaker"] for item in cluster_metadata],
        "start": [item["start"] for item in cluster_metadata],
        "end": [item["end"] for item in cluster_metadata],
        "is_verified": [item.get("is_verified", False) for item in cluster_metadata]
    })
    
    # Save to CSV
    df.to_csv(GLOBAL_MAPPING_FILE, index=False)
    console.print(f"[green]Global speaker mapping saved to {GLOBAL_MAPPING_FILE}[/green]")
    
    # Also create a summary file
    summary_file = os.path.join(SCRIPT_DIR, "speaker_summary.csv")
    
    # Summarize number of segments per speaker per file
    summary = df.groupby(["file", "global_speaker"]).size().reset_index(name="segments")
    summary.to_csv(summary_file, index=False)
    console.print(f"[green]Speaker summary saved to {summary_file}[/green]")
    
    return df

# Update existing mappings with global speaker IDs
def update_mappings_with_global_ids(global_df):
    """Update the existing mappings.csv with global speaker IDs."""
    if not os.path.exists(MAPPING_FILE):
        console.print(f"[yellow]Warning: Mappings file {MAPPING_FILE} does not exist. Cannot update.[/yellow]")
        return
    
    try:
        # Load existing mappings
        df_mappings = pd.read_csv(MAPPING_FILE)
        
        # Create a mapping from file to global speaker
        file_to_speaker = {}
        for file in global_df["file"].unique():
            # Get the most common global speaker for this file's target speaker
            file_speakers = global_df[global_df["file"] == file]
            if not file_speakers.empty:
                most_common_speaker = file_speakers["global_speaker"].value_counts().idxmax()
                file_to_speaker[file] = most_common_speaker
        
        # Add global speaker column to mappings
        df_mappings["global_speaker"] = df_mappings["wav_file"].map(file_to_speaker)
        
        # Save updated mappings
        df_mappings.to_csv(MAPPING_FILE, index=False)
        console.print(f"[green]Updated {MAPPING_FILE} with global speaker IDs[/green]")
        
    except Exception as e:
        console.print(f"[red]Error updating mappings file: {e}[/red]")

# Main function
def main():
    """Main function to run the cross-file speaker recognition tool."""
    clear_console()
    print_title()
    
    # Setup CUDA and get device
    device, cuda_info = setup_cuda()
    
    # Load Speaker Mapping
    if not os.path.exists(MAPPING_FILE):
        console.print(f"[bold red]Error:[/bold red] Speaker mapping file not found: {MAPPING_FILE}", style="red")
        console.print(f"[yellow]Please run the identify-speaker.py script first to create speaker mappings.[/yellow]")
        return
    
    try:
        df_mapping = pd.read_csv(MAPPING_FILE)
        if 'wav_file' not in df_mapping.columns or 'speaker' not in df_mapping.columns:
            console.print(f"[bold red]Error:[/bold red] Invalid mapping file format. Expected columns 'wav_file' and 'speaker'.", style="red")
            return
    except Exception as e:
        console.print(f"[red]Error: Failed to read {MAPPING_FILE}. {e}[/red]")
        return
    
    console.print(f"[bold green]Loaded speaker mapping for {len(df_mapping)} files.[/bold green]")
    
    # Create file processing list
    files_to_process = []
    
    for _, row in df_mapping.iterrows():
        wav_file = row['wav_file']
        target_speaker = row['speaker']
        
        if pd.isna(target_speaker) or target_speaker == "":
            continue
            
        files_to_process.append((wav_file, target_speaker))
    
    if not files_to_process:
        console.print("[bold yellow]No files to process. All mappings are either empty or already processed.[/bold yellow]")
        return
    
    # Check for existing global mapping
    if os.path.exists(GLOBAL_MAPPING_FILE):
        console.print(f"[bold yellow]Global mapping file already exists: {GLOBAL_MAPPING_FILE}[/bold yellow]")
        user_input = console.input("[bold yellow]Do you want to recreate it? (y/n): [/bold yellow]").strip().lower()
        if user_input != 'y':
            console.print("[yellow]Using existing global mapping file.[/yellow]")
            return
    
    # Configure batch size based on CUDA availability
    batch_size = 16 if cuda_info.get("available", False) else 4
    console.print(f"[cyan]Using batch size of {batch_size} for processing.[/cyan]")
    
    # Load embedding model
    try:
        model, model_type = load_embedding_model(device)
    except Exception as e:
        console.print(f"[bold red]Failed to load embedding model: {e}[/bold red]")
        return
    
    # Process files to extract embeddings
    console.print(f"[bold green]Processing {len(files_to_process)} files for speaker embeddings...[/bold green]")
    
    # Create a progress bar
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Extracting speaker embeddings...", total=len(files_to_process))
        
        # Process files in parallel with proper CUDA memory management
        MAX_THREADS = 1 if device.type == "cuda" else min(8, max(1, multiprocessing.cpu_count() - 1))
        console.print(f"[cyan]Using {MAX_THREADS} thread(s) for processing (optimized for {device.type}).[/cyan]")
        
        successful_files = []
        failed_files = []
        
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = {executor.submit(
                process_file_embeddings, 
                file_data, 
                model, 
                model_type,
                device,
                batch_size
            ): file_data for file_data in files_to_process}
            
            for future in as_completed(futures):
                file_data = futures[future]
                try:
                    wav_file, count, status = future.result()
                    
                    # Clear CUDA cache between files if using GPU
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    
                    if status == "Success" or "already exist" in status:
                        successful_files.append((wav_file, count))
                        progress.console.log(f"[green]Processed:[/green] {wav_file} ({count} embeddings)")
                    else:
                        failed_files.append((wav_file, status))
                        progress.console.log(f"[red]Failed:[/red] {wav_file} - {status}")
                except Exception as e:
                    failed_files.append((file_data[0], str(e)))
                    progress.console.log(f"[red]Error:[/red] {file_data[0]} - {e}")
                
                progress.update(task, advance=1)
    
    # Clear CUDA cache after all processing
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Load all embeddings
    console.print("[bold cyan]Loading all speaker embeddings for clustering...[/bold cyan]")
    all_embeddings = []
    
    # List all embedding files
    embedding_files = [f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith("_embeddings.pkl")]
    
    if not embedding_files:
        console.print("[bold red]No embedding files found in embeddings directory.[/bold red]")
        return
    
    # Load each embedding file
    for file in embedding_files:
        file_path = os.path.join(EMBEDDINGS_DIR, file)
        try:
            with open(file_path, "rb") as f:
                embeddings = pickle.load(f)
                if embeddings:  # Only add if the file has embeddings
                    all_embeddings.append(embeddings)
        except Exception as e:
            console.print(f"[red]Error loading {file}: {e}[/red]")
    
    if not all_embeddings:
        console.print("[bold red]No valid embeddings found.[/bold red]")
        return
    
    console.print(f"[bold green]Loaded embeddings from {len(all_embeddings)} files.[/bold green]")
    
    # Cluster embeddings
    console.print("[bold cyan]Clustering speaker embeddings across files...[/bold cyan]")
    cluster_results = cluster_embeddings(all_embeddings)
    
    if not cluster_results:
        console.print("[bold red]Clustering failed. No results.[/bold red]")
        return
    
    # Verify clusters
    console.print("[bold cyan]Verifying speaker clusters...[/bold cyan]")
    user_input = console.input("[bold yellow]Do you want to verify speaker clusters? (y/n): [/bold yellow]").strip().lower()
    
    if user_input == 'y':
        verified_results, verified_clusters = verify_clusters(cluster_results)
        
        # Save results
        save_global_mapping(verified_results)
    else:
        # Save results without verification
        save_global_mapping(cluster_results)
    
    # Update existing mappings
    global_df = pd.read_csv(GLOBAL_MAPPING_FILE)
    update_mappings_with_global_ids(global_df)
    
    # Clear CUDA cache at the end
    if device.type == "cuda":
        torch.cuda.empty_cache()
        console.print(f"[cyan]Final CUDA memory usage: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB[/cyan]")
    
    console.print("[bold green]Cross-file speaker recognition complete![/bold green]")
    console.print(f"[bold green]Results saved to {GLOBAL_MAPPING_FILE}[/bold green]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Program interrupted by user. Exiting...[/bold red]")
        # Clean up CUDA resources if interrupted
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred: {e}[/bold red]")
    finally:
        # Make sure we clean up sounddevice and CUDA resources
        try:
            sd.stop()
        except:
            pass
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass