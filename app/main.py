"""Written by Arman Aydemir. This file contains the main API code for the backend."""
from __future__ import annotations

import logging
import re
from typing import Annotated

from fastapi import Body, Depends, FastAPI, HTTPException, Security, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader

from app.bot import anthropic_bot, openai_bot, openai_bot_stream
from app.db import (
    admin_check,
    api_key_check,
    fetch_session,
    load_bot,
    load_session,
    set_session_to_bot,
    store_bot,
    store_conversation,
)
from app.milvusdb import (
    SESSION_DATA,
    crawl_upload_site,
    delete_expr,
    file_upload,
    session_source_summaries,
    session_upload_ocr,
)
from app.models import (
    BotRequest,
    ChatBySession,
    ChatRequest,
    EngineEnum,
    FetchSession,
    InitializeSession,
    OpinionSearchRequest,
    get_uuid_id,
)
from app.opinion_search import add_opinion_summary, count_opinions, opinion_search
from app.prompts import FILTERED_CASELAW_PROMPT  # noqa: TCH001

# init logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="/var/log/opb.log",
)

logger = logging.getLogger(__name__)

# this is to ensure tracing with langfuse
# @asynccontextmanager
# async def lifespan(app: FastAPI):  # noqa: ARG001, ANN201, D103
#     # Operation on startup

#     yield  # wait until shutdown

#     # Flush all events to be sent to Langfuse on shutdown and
#     # terminate all Threads gracefully. This operation is blocking.
#     langfuse.flush()

X_API_KEY = APIKeyHeader(name="X-API-Key")


def api_key_auth(x_api_key: str = Depends(X_API_KEY)) -> str:
    """Authenticate API key. Source: https://stackoverflow.com/questions/67942766/fastapi-api-key-as-parameter-secure-enough.

    Parameters
    ----------
    x_api_key : str, optional
        api key string, by default Depends(X_API_KEY)

    Raises
    ------
    HTTPException
        if the API key is invalid

    """
    if not api_key_check(x_api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key. Check that you are passing a 'X-API-Key' on your header.",
        )
    return x_api_key



def process_chat_stream(r: ChatRequest):
    bot = load_bot(r.bot_id)
    if bot is None:
        return {"message": "Failure: No bot found with bot id: " + r.bot_id}

    if(bot.chat_model.engine != EngineEnum.openai):
        raise Exception("Invalid bot engine for streaming")

    return openai_bot_stream(r, bot)

def process_chat(r: ChatRequest) -> dict:
    # try:
    bot = load_bot(r.bot_id)
    if bot is None:
        return {"message": "Failure: No bot found with bot id: " + r.bot_id}

    match bot.chat_model.engine:
        case EngineEnum.openai:
            output = openai_bot(r, bot)
        case EngineEnum.anthropic:
            output = anthropic_bot(r, bot)
        case _:
            return {"message": f"Failure: invalid bot engine {bot.chat_model.engine}"}

    # store conversation (and also log the api_key)
    store_conversation(r, output)

    # except Exception as error:
    #     error.
    #     return {"message": "Failure: Internal Error: " + str(error)}
    # return the chat and the bot_id
    return {"message": "Success", "output": output, "bot_id": r.bot_id}


api = FastAPI()


@api.get("/", tags=["General"])
def read_root() -> dict:
    """Just a simple message to check if the API is alive."""
    return {"message": "API is alive"}


@api.post("/invoke_bot", tags=["History Chat"])
def chat(
        request: Annotated[
            ChatRequest,
            Body(
                openapi_examples={
                    "call a bot using history": {
                        "summary": "call a bot using history",
                        "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the bot_id which was "  # noqa: E501
                                       "used}",
                        "value": {
                            "history": [{"role": "user", "content": "hi"}],
                            "bot_id": "custom_4o_dynamic",
                            "api_key": "xyz",
                        },
                    },
                },
            ),
        ],
        api_key: str = Security(api_key_auth)) -> dict:
    """Call a bot with history (only for backwards compat, could be deprecated)."""
    request.api_key = api_key
    return process_chat(request)


@api.post("/initialize_session_chat", tags=["Init Session"])
def init_session(
        request: Annotated[
            InitializeSession,
            Body(
                openapi_examples={
                    "init session": {
                        "summary": "initialize a session",
                        "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the bot_id which was "  # noqa: E501
                                       "used, session_id: the session_id which was created",
                        "value": {
                            "message": "hi, I need help",
                            "bot_id": "some bot id",
                        },
                    },
                },
            ),
        ],
        api_key: str = Security(api_key_auth)) -> dict:
    """Initialize a new session with a message."""
    request.api_key = api_key

    session_id = get_uuid_id()
    set_session_to_bot(session_id, request.bot_id)
    cr = ChatRequest(
        history=[{"role": "user", "content": request.message}],
        bot_id=request.bot_id,
        session_id=session_id,
        api_key=request.api_key,
    )
    response = process_chat(cr)
    try:
        return {
            "message": "Success",
            "output": response["output"],
            "bot_id": request.bot_id,
            "session_id": session_id,
        }
    except:
        return response

@api.post("/initialize_session_chat_stream", tags=["Init Session"], response_model=str)
def init_session_stream(
        request: Annotated[
            InitializeSession,
            Body(
                openapi_examples={
                    "init session": {
                        "summary": "initialize a session",
                        "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the bot_id which was "  # noqa: E501
                                       "used, session_id: the session_id which was created",
                        "value": {
                            "message": "hi",
                            "bot_id": "custom_4o_dynamic",
                            "api_key": "xyz",
                        },
                    },
                },
            ),
        ],
        api_key: str = Security(api_key_auth)) -> dict:
    """Initialize a new session with a message."""
    request.api_key = api_key

    session_id = get_uuid_id()
    set_session_to_bot(session_id, request.bot_id)
    cr = ChatRequest(
        history=[{"role": "user", "content": request.message}],
        bot_id=request.bot_id,
        session_id=session_id,
        api_key=request.api_key,
    )
    logger.info("streaming response here")
    return StreamingResponse(process_chat_stream(cr), media_type="text/event-stream")



@api.post("/chat_session", tags=["Session Chat"])
def chat_session(
        request: Annotated[
            ChatBySession,
            Body(
                openapi_examples={
                    "call a bot using session": {
                        "summary": "call a bot using session",
                        "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the bot_id which was "  # noqa: E501
                                       "used, session_id: the session_id which was used}",
                        "value": {
                            "message": "hi, I need help",
                            "session_id": "some session id",
                        },
                    },
                },
            ),
        ],
        api_key: str = Security(api_key_auth))  -> dict:
    """Continue a chat session with a message."""
    request.api_key = api_key

    cr = load_session(request)
    response = process_chat(cr)
    try:
        return {
            "message": "Success",
            "output": response["output"],
            "bot_id": response["bot_id"],
            "session_id": cr.session_id,
        }
    except:
        return response


@api.post("/fetch_session", tags=["Session Chat"])
def get_session(
        request: Annotated[
            FetchSession,
            Body(
                openapi_examples={
                    "fetch chat history via session": {
                        "summary": "fetch chat history via session",
                        "description": "Returns: {message: 'Success', history: list of messages, bot_id: the bot_id "  # noqa: E501
                                       "which was used, session_id: the session_id which was used}",
                        "value": {
                            "session_id": "some session id",
                        },
                    },
                },
            ),
        ],
        api_key: str = Security(api_key_auth))  -> dict:
    """Fetch the chat history and details of a session."""
    request.api_key = api_key

    cr = fetch_session(request)
    return {
        "message": "Success",
        "history": cr.history,
        "bot_id": cr.bot_id,
        "session_id": cr.session_id,
    }


@api.post("/create_bot", tags=["Bot"])
def create_bot(
        request: Annotated[
            BotRequest,
            Body(
                openapi_examples={
                    "create bot": {
                        "summary": "create opb bot",
                        "description": "Returns: {message: 'Success', bot_id: the new bot_id which was created}",  # noqa: E501
                        "value": {
                            "search_tools": [
                                {
                                    "name": "government-search",
                                    "method": "serpapi",
                                    "prefix": "site:*.gov | site:*.edu | site:*scholar.google.com",
                                    "prompt": "Use to answer questions or find resources about "
                                              "government and laws.",
                                },
                                {
                                    "name": "case-search",
                                    "method": "courtlistener",
                                    "prompt": FILTERED_CASELAW_PROMPT,
                                },
                            ],
                            "vdb_tools": [
                                {
                                    "name": "uscode-query",
                                    "collection_name": "USCode",
                                    "k": 4,
                                    "prompt": "Use to find information about federal laws and regulations.",
                                },
                            ],
                            "chat_model": {
                                "engine": "openai",
                                "model": "gpt-3.5-turbo-0125",
                            },
                        },
                    },
                    "full descriptions of every parameter": {
                        "summary": "Description and Tips",
                        "description": "full descriptions",
                        "value": {
                            "system_prompt": "prompt to use for the bot, replaces the default prompt",
                            "message_prompt": "prompt to use for the bot, this is appended for each message, default is none",  # noqa: E501
                            "search_tools": [
                                {
                                    "name": "name for tool",
                                    "method": "which search method to use, must be one of: serpapi, dynamic_serpapi, "  # noqa: E501
                                              "google, courtlistener",
                                    "prefix": "where to put google search syntax to filter or whitelist results, "  # noqa: E501
                                              "but is also just generally a prefix to add to query passed to tool by "  # noqa: E501
                                              "llm",
                                    "prompt": "description for agent to know when to use the tool",
                                },
                            ],
                            "vdb_tools": [
                                {
                                    "name": "name for tool",
                                    "collection_name": "name of database to query, must be one of: USCode, NCGeneralStatutes, CAP, courtlistener",  # noqa: E501
                                    "k": "the number of text chunks to return when querying the database",
                                    "prompt": "description for agent to know when to use the tool",
                                },
                            ],
                            "chat_model": {
                                "engine": "which library to use for model calls, must be one of: langchain, openai, hive, anthropic, huggingface. "  # noqa: E501
                                      "Default is openai.",
                                "model": "model to be used, openai models work on langchain and openai engines, default is gpt-3.5-turbo-0125",  # noqa: E501
                            },
                        },
                    },
                },
            ),
        ],
        api_key: str = Security(api_key_auth)) -> dict:
    """Create a new bot."""
    request.api_key = api_key

    bot_id = get_uuid_id()
    store_bot(request, bot_id)

    return {"message": "Success", "bot_id": bot_id}


@api.post("/upload_file", tags=["User Upload"])
def upload_file(file: UploadFile, session_id: str, summary: str | None = None,
                api_key: str = Security(api_key_auth)) -> dict:
    """File upload by user.

    Parameters
    ----------
    file : UploadFile
        file to upload.
    session_id : str
        the session to associate the file with.
    summary: str, optional
        A summary of the file written by the user, by default None.

    Returns
    -------
    dict
        Success or failure message.

    """
    logger.info("api_key %s uploading file", api_key)
    try:
        return file_upload(file, session_id, summary)
    except Exception as error:
        return {"message": "Failure: Internal Error: " + str(error)}


@api.post("/upload_files", tags=["User Upload"])
def upload_files(files: list[UploadFile],
    session_id: str, summaries: list[str] | None = None,
    api_key: str = Security(api_key_auth)) -> dict:
    """Upload multiple files by user.

    Parameters
    ----------
    files : list[UploadFile]
        files to upload.
    session_id : str
        the session to associate the file with.
    summaries : list[str] | None, optional
        summaries given by the user, by default None

    Returns
    -------
    dict
        Success or failure message.

    """
    logger.info("api_key %s uploading files", api_key)
    if not summaries:
        summaries = [None] * len(files)
    elif len(files) != len(summaries):
        return {
            "message": f"Failure: did not find equal numbers of files and summaries, "
                f"instead found {len(files)} files and {len(summaries)} summaries.",
        }

    failures = []
    for i, file in enumerate(files):
        result = file_upload(file, session_id, summaries[i])
        if result["message"].startswith("Failure"):
            failures.append(
                f"Upload #{i + 1} of {len(files)} failed. "
                f"Internal message: {result['message']}",
            )

    if len(failures) == 0:
        return {"message": f"Success: {len(files)} files uploaded"}
    return {"message": f"Warning: {len(failures)} failures occurred: {failures}"}


@api.post("/upload_file_ocr", tags=["User Upload"])
def vectordb_upload_ocr(file: UploadFile,
        session_id: str, summary: str | None = None,
        api_key: str = Security(api_key_auth)) -> dict:
    """Upload a file by user and use OCR to extract info."""
    logger.info("api_key %s uploading file with OCR", api_key)
    return session_upload_ocr(file, session_id, summary if summary else None)


@api.post("/delete_file", tags=["Vector Database"])
def delete_file(filename: str, session_id: str, api_key: str = Security(api_key_auth)):
    """Delete a file from the sessions database.

    Parameters
    ----------
    filename : str
        filename to delete.
    session_id : str
        session to delete the file from.

    """
    logger.info("api_key %s deleting file %s", api_key, filename)
    return delete_expr(
        SESSION_DATA,
        f"metadata['source']=='{filename}' and session_id=='{session_id}'",
    )


@api.post("/delete_files", tags=["Vector Database"])
def delete_files(filenames: list[str], session_id: str, api_key: str = Security(api_key_auth)) -> dict:
    """Delete multiple files from the database.

    Parameters
    ----------
    filenames : list[str]
        filenames to delete.
    session_id : str
        session to delete the file from

    Returns
    -------
    dict
        Success message with number of files deleted.

    """
    logger.info("api_key %s deleting files", api_key)
    for filename in filenames:
        delete_file(filename, session_id)
    return {"message": f"Success: deleted {len(filenames)} files"}


@api.post("/get_session_files", tags=["Vector Database"])
def get_session_files(session_id: str, api_key: str = Security(api_key_auth)) -> dict:
    """Get names of all files associated with a session.

    Parameters
    ----------
    session_id : str
        session to get files from.

    Returns
    -------
    dict
        Success message with list of filenames.

    """
    logger.info("api_key %s getting session files for session %s", api_key, session_id)
    source_summaries = session_source_summaries(session_id)
    files = list(source_summaries.keys())
    return {"message": f"Success: found {len(files)} files", "result": files}


@api.post("/delete_session_files", tags=["Vector Database"])
def delete_session_files(session_id: str, api_key: str = Security(api_key_auth)):
    """Delete all files associated with a session.

    Parameters
    ----------
    session_id : str
        _description_

    Returns
    -------
    _type_
        _description_
    """
    logger.info("api_key %s deleting session files for session %s", api_key, session_id)
    return delete_expr(SESSION_DATA, f"session_id=='{session_id}'")


@api.post("/upload_site", tags=["Admin Upload"])
def vectordb_upload_site(site: str, collection_name: str,
        description: str, api_key: str = Security(api_key_auth)):
    if not admin_check(api_key):
        return {"message": "Failure: API key invalid"}
    return crawl_upload_site(collection_name, description, site)


@api.post("/search_opinions", tags=["Opinion Search"])
def search_opinions(
    req: OpinionSearchRequest,
    api_key: str = Security(api_key_auth),
) -> dict:
    if not api_key_check(api_key):
        return {"message": "Failure: API key invalid"}
    try:
        results = opinion_search(req)
    except Exception as error:
        return {"message": "Failure: Internal Error: " + str(error)}
    else:
        return {"message": "Success", "results": results}


@api.get("/get_opinion_summary", tags=["Opinion Search"])
def get_opinion_summary(
    opinion_id: int,
    api_key: str = Security(api_key_auth),
) -> dict:
    if not api_key_check(api_key):
        return {"message": "Failure: API key invalid"}
    try:
        summary = add_opinion_summary(opinion_id)
    except Exception as error:
        return {"message": "Failure: Internal Error: " + str(error)}
    else:
        return {"message": "Success", "result": summary}

@api.get("/get_opinion_count", tags=["Opinion Search"])
def get_opinion_count(api_key: str = Security(api_key_auth)) -> dict:
    if not api_key_check(api_key):
        return {"message": "Failure: API key invalid"}
    return {"message": "Success", "opinion_count": count_opinions()}
