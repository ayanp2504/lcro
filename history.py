from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_openai import ChatOpenAI
from backend_lcro.retrieval_graph.graph import graph
from typing import Any
import logging
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)

load_dotenv('.config')  # Load environment variables from the .config file

async def run_graph(inputs: dict[str, Any]) -> dict[str, Any]:
    results = await graph.ainvoke(
        {
            "messages": [("human", inputs["question"])],
        }
    )
    return results

# Function to contextualize a question based on session chat history
def contextualize_question(new_question: str, session_id: str, model: ChatOpenAI, db_path: str) -> str:
    """Reformulates a new question based on the chat history for the given session."""

    # Retrieve chat history for the given session
    history_store = SQLChatMessageHistory(session_id=session_id, connection=f"sqlite:///{db_path}")
    chat_history = history_store.get_messages()

    

    logging.info("Chat history:", chat_history)
    
    # If the chat history is empty, return the question as is
    if not chat_history:
        history_store.add_message( HumanMessage(
        content=new_question
    ))
        return new_question
    
    print("Chat History before trim", chat_history)
    chat_history = trim_messages(
    chat_history,
    strategy="last",
    token_counter=ChatOpenAI(model="gpt-4"),
    max_tokens=3000,
    start_on=("ai", "human"),
    end_on=("ai", "tool"),
    include_system=True,
    allow_partial=True,
    )
    history_store.add_message( HumanMessage(
        content=new_question
    ))
    # Define the system prompt for contextualizing the question
    contextualize_q_system_prompt = """
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. ## Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    """
    
    # Set up the prompt template with placeholders for chat history and the new question
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )

    # Format the prompt with chat history and new question
    prompt_with_history = contextualize_q_prompt.format(chat_history=chat_history, input=new_question)
    
    # Invoke the model with the prompt
    contextualized_question = model.invoke(prompt_with_history)
    
    return contextualized_question.content

# Usage Example:
model = ChatOpenAI()  # Instantiate the OpenAI model
session_id = "intro"
db_path = r"D:\Excellerate LLM\App\backend_lcro\databases\chat_history.db"
new_question = "What is in the image?"

# Call the function to contextualize the question
contextualized_question = contextualize_question(new_question, session_id, model, db_path)
result = run_graph({"question": "{contextualized_question}"})
history_store = SQLChatMessageHistory(session_id=session_id, connection=f"sqlite:///{db_path}")
history_store.add_message(result)

# print("Contextualized Question:", contextualized_question)
