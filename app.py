from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, Form
from pydantic import BaseModel, Field, HttpUrl
from .ingest import ingest_docs, delete_docs
from .process_urls import ingest_urls
from .ask import get_chat_history_with_response
from .clear_chat import clear_chat
# from ingest import ingest_docs
# from ask import get_chat_history_with_response
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import argparse
from .ask_audio import start_recording, get_transcription, process_audio_file
app = FastAPI()

class S3URIRequest(BaseModel):
    s3_uri: str = Field(..., min_length=1, description="The S3 URI must not be empty")

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The query must not be empty")
    session_id : str = Field(..., min_length=1, description="The session id must not be empty")
    doc_list: List[str] = Field(..., min_items=1, description="List of document sources to be used for filtering; must contain at least one item")

class ClearChatSession(BaseModel):
    session_id : str = Field(..., min_length=1, description="The session id must not be empty")
    
#  Request Model for Start/Stop Action
class RecordingActionRequest(BaseModel):
    action: str = Field(..., min_length=1, description="Action should be either 'start' or 'stop'")
    session_id : str = Field(..., min_length=1, description="The session id must not be empty")
    doc_list: List[str] = Field(..., min_items=1, description="List of document sources to be used for filtering; must contain at least one item")
# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers (including custom ones)
)



@app.post("/ingest-docs/")
async def ingest_documents(request: S3URIRequest):
    """Endpoint to ingest documents from an S3 URI."""
    if not request.s3_uri:
        raise HTTPException(status_code=400, detail="The S3 URI must not be empty.")
    
    result = ingest_docs(request.s3_uri)
    return {"message": f"Successfully processed {result} documents."}

@app.post("/ingest-urls/")
async def ingest_urls_endpoint(https_urls: List[HttpUrl]):
    """Endpoint to ingest documents from a list of HTTPS URLs."""
    if not https_urls:
        raise HTTPException(status_code=400, detail="The web URLs must not be empty.")
    
    url_list = [str(url) for url in https_urls]
    
    # Call the ingest_urls function with the provided list
    result =  ingest_urls(url_list)
    return result

@app.post("/delete-docs/")
async def delete_documents(request: S3URIRequest):
    """Endpoint to delete documents from an S3 URI."""
    if not request.s3_uri:
        raise HTTPException(status_code=400, detail="The S3 URI must not be empty.")
    
    result = delete_docs(request.s3_uri)
    return result

@app.post("/ask/")
async def ask_question(request: AskRequest):
    """Endpoint to ingest documents from an S3 URI."""
    if not request.query:
        raise HTTPException(status_code=400, detail="The S3 URI must not be empty.")
    
    result = await get_chat_history_with_response(request.query, request.session_id, request.doc_list)
    return result

@app.post("/clear-chat/")
async def clear_chat_session(request: ClearChatSession):
    """Endpoint to clear the chat history for a given session."""
    # Call the clear_chat function with the provided session ID
    await clear_chat(request.session_id)
    return {"message": "Chat history cleared."}

# Endpoint to start/stop recording based on the action
@app.post("/recording")
async def handle_recording(action_request: RecordingActionRequest, background_tasks: BackgroundTasks):
    """Handles recording: starts recording if action is 'start', stops and returns transcription if action is 'stop'."""
    if action_request.action == "start":
        try:
            # Start recording
            start_recording()  # Call function from audio_processing.py
            return {"message": "Recording started, press Enter to stop it."}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    elif action_request.action == "stop":
        try:
            # Stop recording and return transcription
            transcription = get_transcription()  # Call function from audio_processing.py
            result = await get_chat_history_with_response(transcription, action_request.session_id, action_request.doc_list)
            return result
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    else:
        raise HTTPException(status_code=400, detail="Invalid action. Use 'start' or 'stop'.")


@app.post("/voice_assistant")
async def handle_recording(
    audiofile: UploadFile,  # File input
    session_id: str = Form(...),  # Form field for session ID
    doc_list: List[str] = Form(...),  # Form field for list of documents
):
    """
    Handles recording: processes the uploaded audio file and returns transcription and chat history.
    """
    try:
        # Read the audio file into memory
        audio_data = await audiofile.read()
        print(type(audio_data))
        
        # Process audio data and get transcription
        transcription = process_audio_file(audio_data)  # Update this to your new get_transcription logic
        
        # Get chat history with response
        result = await get_chat_history_with_response(transcription, session_id, doc_list)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="172.31.11.168")
    parser.add_argument("--port", type=int, default=90)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
