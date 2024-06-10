"""Written by Arman Aydemir. Used to access and store data in the Firestore database."""
import os
from json import loads
from typing import Optional

import firebase_admin
from firebase_admin import credentials, firestore

from app.models import (
    BotRequest,
    ChatBySession,
    ChatRequest,
    EncoderParams,
    FetchSession,
    MilvusMetadataEnum,
    get_uuid_id,
)

# langfuse = Langfuse()

# sdvlp session
# 1076cca8-a1fa-415a-b5f8-c11da178d224

# which version of db we are using
VERSION = "_vj1"
BOT_COLLECTION = "bots"
MILVUS_COLLECTION = "milvus"
CONVERSATION_COLLECTION = "conversations"

firebase_config = loads(os.environ["Firebase"])
cred = credentials.Certificate(firebase_config)
firebase_admin.initialize_app(cred)
db = firestore.client()

def api_key_check(api_key: str) -> bool:
    """Check if api key is valid.

    Parameters
    ----------
    api_key : str
        the api key

    Returns
    -------
    bool
        true if valid

    """
    return db.collection("api_keys").document(api_key).get().exists


def admin_check(api_key: str) -> bool:
    """Check if api key is valid admin key.

    Parameters
    ----------
    api_key : str
        the api key

    Returns
    -------
    bool
        true if valid admin key

    """
    result = db.collection("api_keys").document(api_key).get()
    if result.exists:
        return result.to_dict()["admin"]
    return False

def store_conversation(r: ChatRequest, output: str) -> bool:
    """Store the conversation in the database.

    Args:
    ----
        r (ChatRequest): Chat request object
        output (str): The output from the bot

    Returns:
    -------
        bool: True if successful, False otherwise

    """
    if r.session_id is None or r.session_id == "":
        r.session_id = get_uuid_id()

    data = r.model_dump()
    data["num_msg"] = len(r.history)

    t = firestore.SERVER_TIMESTAMP
    data["timestamp"] = t

    db.collection(CONVERSATION_COLLECTION + VERSION).document(r.session_id).set(data)

    db.collection(CONVERSATION_COLLECTION + VERSION).document(r.session_id).set(
        {"last_message_timestamp": t}, merge=True)

    return True


def set_session_to_bot(session_id: str, bot_id: str) -> bool:
    """Set the session to use the bot.

    Args:
    ----
        session_id (str): the session uuid
        bot_id (str): the bot uuid

    Returns:
    -------
        bool: True if successful, False otherwise

    """
    db.collection(CONVERSATION_COLLECTION + VERSION).document(session_id).set(
        {"bot_id": bot_id}, merge=True)
    return True


def load_session(r: ChatBySession) -> ChatRequest:
    """Load the associated session to continue the conversation.

    Args:
    ----
        r (ChatBySession): Obj containing the session id and message from user

    Returns:
    -------
        ChatRequest: The chat request object with appended user message

    """
    session_data = (
        db.collection(CONVERSATION_COLLECTION + VERSION)
        .document(r.session_id).get()
    ) 

    session_data = session_data.to_dict()

    #add user message to history
    history = session_data["history"]
    history.append({"role": "user", "content": r.message})


    return ChatRequest(
        history=history,
        bot_id=session_data["bot_id"],
        session_id=r.session_id,
        api_key=r.api_key,
    )


def fetch_session(r: FetchSession) -> ChatRequest:
    """Load the associated session data from database.

    Args:
    ----
        r (FetchSession): Obj containing the session data

    Returns:
    -------
        ChatRequest: The chat request object with data from firestore

    """
    session_data = (
        db.collection(CONVERSATION_COLLECTION + VERSION)
        .document(r.session_id).get()
    ) #loading session data from db

    session_data = session_data.to_dict()

    return ChatRequest( #building the actual ChatRequest object
        history=session_data["history"],
        bot_id=session_data["bot_id"],
        session_id=r.session_id,
        api_key=r.api_key,
    )

def store_bot(r: BotRequest, bot_id: str) -> bool:
    """Store the bot in the database.

    Parameters
    ----------
    r : BotRequest
        The bot object to store.
    bot_id : str
        The bot id to use.

    Returns
    -------
    bool
        True if successful, False otherwise.

    """
    data = r.model_dump()
    data["timestamp"] = firestore.SERVER_TIMESTAMP
    db.collection(BOT_COLLECTION + VERSION).document(bot_id).set(data)


def load_bot(bot_id: str) -> BotRequest:
    """Load the bot from the database using the bot_id.

    Parameters
    ----------
    bot_id : str
        The bot id to load.

    Returns
    -------
    BotRequest
        The bot object.

    """
    bot = db.collection(BOT_COLLECTION + VERSION).document(bot_id).get()
    if bot.exists:
        return BotRequest(**bot.to_dict())

    return None

def load_vdb(collection_name: str) -> dict:
    """Load the parameters for a collection from the database.

    Parameters
    ----------
    collection_name : str
        The name of the collection that uses the parameters.

    Returns
    -------
    dict
        The collection parameters: encoder, metadata_format, fields.

    """
    data = db.collection(MILVUS_COLLECTION).document(collection_name).get()
    if data.exists:
        return data.to_dict()

    return None

def store_vdb(
    collection_name: str,
    encoder: EncoderParams,
    metadata_format: MilvusMetadataEnum,
    fields: Optional[list] = None,
) -> bool:
    """Store the configuration of a Milvus collection in the database.

    Parameters
    ----------
    collection_name : str
        The collection that uses the configuration.
    encoder : EncoderParams
        The EncoderParams object to store.
    metadata_format : MilvusMetadataEnum
        The MilvusMetadataEnum object to store.
    fields : list
        The list of field names to store if metadata_format is field.

    Returns
    -------
    bool
        True if successful, False otherwise.

    """
    data = {
        "encoder": encoder.model_dump(),
        "metadata_format": metadata_format,
        "timestamp": firestore.SERVER_TIMESTAMP,
    }
    if fields is not None:
        data["fields"] = fields
    db.collection(MILVUS_COLLECTION).document(collection_name).set(data)
    return True
