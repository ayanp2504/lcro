from backend_lcro.constants import CHAT_HISTORY_DB_PATH
from langchain_community.chat_message_histories import SQLChatMessageHistory

# Function to clear the chat history for a given session
async def clear_chat(session_id: str) -> None:
    """Clears the chat history for the given session."""
    history_store = SQLChatMessageHistory(session_id=session_id, connection=CHAT_HISTORY_DB_PATH)
    history_store.clear()