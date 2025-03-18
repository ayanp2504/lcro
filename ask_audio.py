import sounddevice as sd
import numpy as np
import io
from scipy.io.wavfile import write
import openai
import threading
from openai import OpenAI

# Initialize OpenAI client
openai_client = OpenAI()
# Global configuration for recording
sample_rate = 16000  # Sampling rate for audio
audio_data = []      # List to store audio chunks
recording = False    # Flag to control recording


def record_audio():
    """Continuously records audio until the recording flag is set to False."""
    global audio_data, recording
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
        print("Recording started...")
        while recording:
            audio_chunk, _ = stream.read(1024)  # Read audio data in chunks
            audio_data.append(audio_chunk)


def stop_recording():
    """Sets the recording flag to False."""
    global recording
    recording = False
    print("Recording stopped.")


def start_recording():
    """Starts the audio recording in a separate thread."""
    global audio_data, recording
    if recording:
        raise Exception("Recording is already in progress.")
    
    audio_data = []  # Clear any previous audio data
    recording = True  # Set the flag to start recording
    
    # Start the recording process in a separate thread
    threading.Thread(target=record_audio, daemon=True).start()


def get_transcription():
    """Stops the recording, processes the audio, and returns the transcription."""
    global audio_data, recording
    if not recording:
        raise Exception("No recording is in progress.")
    
    stop_recording()  # Stop the recording process
    
    # Combine all audio chunks into a single numpy array
    audio_data_combined = np.concatenate(audio_data, axis=0)

    # Convert to WAV format in-memory
    audio_bytes = io.BytesIO()
    write(audio_bytes, sample_rate, audio_data_combined)  # Use scipy's write function to save to BytesIO
    audio_bytes.seek(0)  # Go to the start of the BytesIO buffer
    audio_bytes.name = "audio.wav"  # Set a filename for the in-memory file

    # Transcribe via Whisper
    transcription = openai_client.audio.transcriptions.create(
       model="whisper-1", 
       file=audio_bytes,
    )

    # Return the transcription
    return transcription.text

def process_audio_file(audiofile):
    """Processes the uploaded audio file and returns the transcription."""

    # Convert to WAV format in-memory
    audio_bytes = io.BytesIO(audiofile)

    audio_bytes.seek(0)  # Go to the start of the BytesIO buffer
    audio_bytes.name = "audio.wav"  # Set a filename for the in-memory file

    # Transcribe via Whisper
    transcription = openai_client.audio.transcriptions.create(
       model="whisper-1", 
       file=audio_bytes,
    )

    # Return the transcription
    return transcription.text