import asyncio
import websockets
import whisper
import subprocess
from datetime import datetime
import json
from openai import OpenAI

# Set your OpenAI API key
client = OpenAI(
    api_key="private",
)

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

    # Transcribe audio using Whisper
    transcript = await transcribe_audio(wav_file)
    
    # Summarize the transcript using OpenAI
    if transcript:
        summary = await summarize_transcript(transcript)
        response = {"text": transcript, "summary": summary}
        try:
            await websocket.send(json.dumps(response))
            print("Response sent")
        except websockets.ConnectionClosed:
            print("Connection closed before sending response")

async def transcribe_audio(audio_file):
    try:
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        print("Model loaded, transcribing...")
        result = model.transcribe(audio_file)
        transcript = result["text"]
        print("Transcription: ", transcript)
        return transcript
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

async def summarize_transcript(transcript):
    try:
        print("Summarizing transcript using OpenAI API...") 
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": f"""
Conversation: Identify the speakers from the transcript and label them as 'Server' or 'sales rep' and 'Customer' with their sentences. 
Summarize: Summarize the transcript.
Intent: What is the intent of the customer in the transcript?
Sentiment: What is the sentiment of the parties involved?

Transcript:

{transcript}

Identifying Speakers and Labeling Sentences:
"""
                }
                ]
        )       
        summary = response.choices[0].message.content.strip()
        print("Conversation Analysis:\n\n\n ", summary)
        return summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        return None

async def main():
    print("Starting WebSocket server...")
    async with websockets.serve(audio_handler, "localhost", 8000):
        print("WebSocket server listening on ws://localhost:8000/")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
