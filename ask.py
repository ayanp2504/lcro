from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from fastapi.responses import JSONResponse
import asyncio
from dotenv import load_dotenv
from typing import Any, List
from backend_lcro.retrieval_graph.graph import graph
import logging
from langchain_core.messages import trim_messages
from backend_lcro.constants import CHAT_HISTORY_DB_PATH

load_dotenv('.config')  # Load environment variables from the .config file

db_path = CHAT_HISTORY_DB_PATH
model = ChatOpenAI(model='gpt-4o')

def run_graph(inputs: dict[str, Any], doc_list: List[str], user_id: str) -> dict[str, Any]:
    config = {"configurable": {"user_id": user_id,
                                "search_kwargs": 
                               {"k": 2, 
                                'filter': {
                    'filename': {'$in': doc_list}  # Use the user-provided document list
                }}}}
    
    results = graph.ainvoke(
        {
            "messages": [("human", inputs["question"])],
            
        },
        config=config
        
    )
    return results

# Function to contextualize a question based on session chat history
async def contextualize_question(new_question: str, session_id: str, model: ChatOpenAI) -> str:
    """Reformulates a new question based on the chat history for the given session."""
    history_store = SQLChatMessageHistory(session_id=session_id, connection=CHAT_HISTORY_DB_PATH)
    chat_history = history_store.get_messages()

    logging.info("Chat history:", chat_history)
    
    if not chat_history:
        return new_question

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
    
    

    contextualize_q_system_prompt = """
    Given a chat history and the latest user question, which may or may not reference context from the chat history, formulate a standalone question if it refers to chat history such it that can be understood without the chat history. 
    This standalone question will be used to retrieve relevant document chunks from a vector database. 
    Do NOT answer the question; simply reformulate it if needed, or return it as is.

    ** Chat History Starts **
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    prompt_with_history = contextualize_q_prompt.format(chat_history=chat_history, 
                                                        input="\n** Chat History Ends **\nLatest User Question:" + new_question)
    
    contextualized_question = await model.ainvoke(prompt_with_history)
    return contextualized_question.content

async def get_chat_history_with_response(new_question: str, session_id: str, doc_list: List[str]) -> List[HumanMessage]:
    """Returns the updated chat history after adding the latest question and generating a response."""
    
    contextualized_question = await contextualize_question(new_question, session_id, model)
    result = await run_graph({"question": contextualized_question}, doc_list, user_id=session_id)
    
    imagess3 = result['documents']
    print(len(imagess3), "images3")

    image_s3_uris = []
    for doc in imagess3:
        if doc.metadata['doc_type'] == 'pdf_image':
            image_s3_uris.append(doc.metadata['extracted_image_s3_uri'])
        elif doc.metadata['doc_type'] == 'image':
            image_s3_uris.append(doc.metadata['s3_uri'])

    result = result['answer']
    


    history_store = SQLChatMessageHistory(session_id=session_id, connection=CHAT_HISTORY_DB_PATH)
    history_store.add_message(HumanMessage(content=new_question))
    history_store.add_message(AIMessage(content=result))
    
    # Return the updated chat history
    # hist_msg =  history_store.get_messages()
    # msg_list = []
    # for msg in hist_msg:
    #     if isinstance(msg, HumanMessage):
    #         msg_list.append(["human", msg.content])
    #     elif isinstance(msg, AIMessage):
    #         msg_list.append(["ai", msg.content])

    msg_list = []
    msg_list.append(["human", new_question])
    msg_list.append(["ai", result])
    
    return JSONResponse(content={
            "chat_history": msg_list,
            "session_id": session_id,  # Return session ID to the client for future requests
            "image_s3":image_s3_uris
        })

# Example usage
if __name__ == "__main__":
    session_id = "intro"
    new_question = "ANy significant trials?"

    updated_history = asyncio.run(get_chat_history_with_response(new_question, session_id))
    print("Updated Chat History:", updated_history)
