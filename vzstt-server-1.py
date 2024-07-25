import asyncio
import websockets
import whisper
from pyannote.audio import Pipeline
import os
import subprocess
from datetime import datetime
import shutil

# Replace 'YOUR_HF_ACCESS_TOKEN' with your actual Hugging Face access token
HF_ACCESS_TOKEN = 'token'

async def audio_handler(websocket, path):
    print("Client connected")
    audio_frames = []
    while True:
        try:
            data = await websocket.recv()
            audio_frames.append(data)
            print(f"Received audio data chunk of size: {len(data)}")
        except websockets.ConnectionClosed:
            print("Connection closed")
            break

    if not audio_frames:
        print("No audio frames received")
        return

    # Generate unique filenames using timestamps
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    webm_file = f'received_audio_{timestamp}.webm'
    wav_file = f'received_audio_{timestamp}.wav'

    # Save received audio to a WebM file
    try:
        with open(webm_file, 'wb') as f:
            f.write(b''.join(audio_frames))
        print(f"Audio saved to {webm_file}")
    except Exception as e:
        print(f"Error saving audio file: {e}")
        return

    # Convert WebM to WAV using FFmpeg
    try:
        print(f"Converting {webm_file} to {wav_file} using FFmpeg...")
        result = subprocess.run(
            ['ffmpeg', '-i', webm_file, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '44100', wav_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30  # Timeout after 30 seconds
        )
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr.decode()}")
            return
        print(f"Converted {webm_file} to {wav_file}")
    except subprocess.TimeoutExpired:
        print(f"FFmpeg conversion timed out.")
        return
    except Exception as e:
        print(f"Error converting audio file: {e}")
        return

    # Check if the WAV file exists and print its size
    try:
        with open(wav_file, 'rb') as f:
            data = f.read()
            print(f"WAV file size: {len(data)} bytes")
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        return

    # Transcribe and diarize audio
    asyncio.create_task(transcribe_and_diarize(wav_file))

async def transcribe_and_diarize(audio_file):
    try:
        # Transcribe audio using Whisper
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        print("Model loaded, transcribing...")
        result = model.transcribe(audio_file)
        print("Transcription: ", result["text"])
    except Exception as e:
        print(f"Error during transcription: {e}")
        return

    try:
        # Perform diarization
        print("Loading diarization pipeline...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_ACCESS_TOKEN)
        print("Pipeline loaded, performing diarization...")
        diarization = pipeline(audio_file)
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f"Start: {turn.start:.1f}s End: {turn.end:.1f}s Speaker: {speaker}")
        print("Diarization completed successfully.")
    except Exception as e:
        print(f"Error during diarization: {e}")
        if 'required privilege is not held by the client' in str(e):
            try:
                print("Attempting to manually move the file with elevated privileges...")
                src = r"C:\Users\SivaM\.cache\huggingface\hub\models--speechbrain--spkrec-ecapa-voxceleb\snapshots\eac27266f68caa806381260bd44ace38b136c76a\hyperparams.yaml"
                dst = r"C:\Users\SivaM\.cache\torch\pyannote\speechbrain\hyperparams.yaml"
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst)
                print("File moved successfully. Retrying diarization...")
                pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_ACCESS_TOKEN)
                diarization = pipeline(audio_file)
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    print(f"Start: {turn.start:.1f}s End: {turn.end:.1f}s Speaker: {speaker}")
                print("Diarization completed successfully.")
            except Exception as move_err:
                print(f"Failed to manually move the file: {move_err}")

async def main():
    print("Starting WebSocket server...")
    async with websockets.serve(audio_handler, "localhost", 8000):
        print("WebSocket server listening on ws://localhost:8000/")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
