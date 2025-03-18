import whisper
from dotenv import load_dotenv

load_dotenv('.config')  # Load environment variables from the .config file

# # Load the Whisper model (choose 'base', 'small', 'medium', or 'large' based on your requirements)
# model = whisper.load_model("base")

# # Path to the MP3 file
# mp3_file = r"C:\\Users\\ayanp\\Downloads\\Imagine Dragons - Believer (Audio).mp3"

# # Transcribe the MP3 file
# print("Transcribing audio...")
# result = model.transcribe(mp3_file)

# # Output the transcription
# print("\nTranscription:")
# print(result["text"])


from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from langchain_community.document_loaders.blob_loaders import Blob

# Initialize the parser
parser = OpenAIWhisperParser(language='en')

# Specify the path to your audio file
audio_path = r"C:\Users\ayanp\Downloads\30october.mp3"
audio_blob = Blob(path=audio_path)

# Parse the audio file
documents = parser.lazy_parse(blob=audio_blob)

print("#"*40)
# Print the transcribed text
for doc in documents:
    print(doc.page_content)
