"""The search api functions and search toolset creation. Written by Arman Aydemir."""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context

import requests
from langchain.agents import Tool
from langfuse.decorators import observe
from serpapi.google_search import GoogleSearch

from app.courtlistener import courtlistener_search, courtlistener_tool_args
from app.milvusdb import query, source_exists, upload_site
from app.models import (
    BotRequest,
    EngineEnum,
    OpinionSearchRequest,
    SearchMethodEnum,
    SearchTool,
)
from app.prompts import FILTERED_CASELAW_PROMPT

GoogleSearch.SERP_API_KEY = os.environ["SERPAPI_KEY"]

COURTROOM5_SEARCH_CX_KEY = "05be7e1be45d04eda"

search_collection = "search_collection_vj1"

def filtered_search(results: dict) -> dict:
    """Filter search results returned by serpapi to only include relevant results.

    Parameters
    ----------
    results : dict
         the results from serpapi search

    Returns
    -------
    dict
        filtered results

    """
    new_dict = {}
    if "sports_results" in results:
        new_dict["sports_results"] = results["sports_results"]
    if "organic_results" in results:
        new_dict["organic_results"] = results["organic_results"]
    return new_dict


@observe()
def dynamic_serpapi_tool(qr: str, prf: str, num_results: int = 5) -> dict:
    """Upgraded serpapi tool, scrape the websites and embed them to query whole pages.

    Parameters
    ----------
    qr : str
        the query
    prf : str
        the prefix given by tool (used for whitelists)
    num_results : int, optional
        number of results to return, by default 5

    Returns
    -------
    dict
        result of the query on the embeddings uploaded to the search collection

    """
    response = filtered_search(
        GoogleSearch({
            "q": prf + " " + qr,
            "num": num_results,
        }).get_dict())

    def process_site(result: dict) -> None:
        try:
            if(not source_exists(search_collection, result["link"])):
                print("Uploading site: " + result["link"])
                upload_site(search_collection, result["link"])
        except Exception as error:
            print("Warning: Failed to upload site for dynamic serpapi: " + result["link"])
            print("The error was: " + str(error))

    with ThreadPoolExecutor() as executor:
        futures = []
        for result in response["organic_results"]:
            ctx = copy_context()
            def task(r=result, context=ctx):  # noqa: ANN001, ANN202
                return context.run(process_site, r)
            futures.append(executor.submit(task))

        for future in as_completed(futures):
            _ = future.result()

    return query(search_collection, qr, k=3)


def google_search_tool(qr: str, prf: str, max_len: int = 6400) -> str:
    """Query the google search api.

    Parameters
    ----------
    qr : str
        the query itself
    prf : str
        the prefix given by the tool (used for whitelists)
    max_len : int, optional
        maximum length of response text, by default 6400

    Returns
    -------
    str
        the search results

    """
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    params = {
        "key": os.environ["GOOGLE_SEARCH_API_KEY"],
        "cx": os.environ["GOOGLE_SEARCH_API_CX"],
        "q": prf + " " + qr,
    }
    return str(
        requests.get("https://www.googleapis.com/customsearch/v1",
                     params=params,
                     headers=headers, timeout=30).json())[0:max_len]


@observe()
def courtroom5_search_tool(qr: str, prf: str="", max_len: int = 6400) -> str:
    """Query the custom courtroom5 google search api.

    Whitelisted sites defined by search cx key.

    Parameters
    ----------
    qr : str
        the query itself
    prf : str
        the prefix given by the tool (whitelisted sites defined by search cx key)
    max_len : int, optional
        maximum length of response text, by default 6400

    Returns
    -------
    str
        the search results

    """
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    params = {
        "key": os.environ["GOOGLE_SEARCH_API_KEY"],
        "cx": COURTROOM5_SEARCH_CX_KEY,
        "q": prf + " " + qr,
    }
    return str(
        requests.get("https://www.googleapis.com/customsearch/v1",
                     params=params,
                     headers=headers, timeout=30).json())[0:max_len]


# Implement this for regular programatic google search as well.
@observe()
def dynamic_courtroom5_search_tool(qr: str, prf: str="") -> dict:
    """Query the custom courtroom5 google search api, scrape the sites and embed them.

    Whitelisted sites defined by search cx key.

    Parameters
    ----------
    qr : str
        the query itself
    prf : str
        the prefix given by the tool

    Returns
    -------
    str
        the search results

    """
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    params = {
        "key": os.environ["GOOGLE_SEARCH_API_KEY"],
        "cx": COURTROOM5_SEARCH_CX_KEY,
        "q": prf + " " + qr,
    }
    response = requests.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params=params,
                    headers=headers,
                    timeout=30,
                ).json()

    def process_site(result: dict) -> None:
        try:
            if(not source_exists(search_collection, result["link"])):
                print("Uploading site: " + result["link"])
                upload_site(search_collection, result["link"])
        except Exception as error:
            print("Warning: Failed to upload site for dynamic serpapi: " + result["link"])
            print("The error was: " + str(error))

    for result in response["items"]:
        process_site(result)

    return query(search_collection, qr)

def serpapi_tool(qr: str, prf: str, num_results: int = 5) -> dict:
    """Query the serpapi search api.

    Parameters
    ----------
    qr : str
        the query
    prf : str
        prefix defined by tool (used for whitelist)
    num_results : int, optional
        number of results to return, by default 5

    Returns
    -------
    dict
        the dict of results

    """
    return filtered_search(
        GoogleSearch({
            "q": prf + " " + qr,
            "num": num_results,
        }).get_dict())


def dynamic_serpapi_tool_creator(t: SearchTool) -> Tool:
    """Generate the dynamic serpapi tool to give to agents.

    Parameters
    ----------
    t : SearchTool
       The SearchTool object which describes the tool

    Returns
    -------
    Tool
       The tool created to be used by agents

    """
    name = t.name
    prompt = t.prompt
    prf = t.prefix

    async def async_search_tool(qr: str, prf: str) -> dict:
        return dynamic_serpapi_tool(qr, prf)

    tool_func = lambda qr: dynamic_serpapi_tool(qr, prf)  # noqa: E731
    co_func = lambda qr: async_search_tool(qr, prf)  # noqa: E731
    return Tool(name=name,
                func=tool_func,
                coroutine=co_func,
                description=prompt)


def search_tool_creator(t: SearchTool) -> Tool:
    """Create a google search api tool for agents to use.

    Parameters
    ----------
    t : SearchTool
        The SearchTool object which describes the tool

    Returns
    -------
    Tool
        The tool created to be used by agents

    """
    name = t.name
    prompt = t.prompt
    txt = t.prefix

    async def async_search_tool(qr: str, txt: str) -> str:
        return google_search_tool(qr, txt)

    tool_func = lambda qr: google_search_tool(qr, txt)  # noqa: E731
    co_func = lambda qr: async_search_tool(qr, txt)  # noqa: E731

    return Tool(name=name,
                func=tool_func,
                coroutine=co_func,
                description=prompt)

def courtroom5_tool_creator(t: SearchTool) -> Tool:
    """Create a custom courtroom5 search api tool for agents to use.

    Parameters
    ----------
    t : SearchTool
        The SearchTool object which describes the tool

    Returns
    -------
    Tool
        The tool created to be used by agents

    """
    name = t.name
    prompt = t.prompt
    txt = t.prefix

    async def async_search_tool(qr: str, txt: str) -> str:
        return courtroom5_search_tool(qr, txt)

    tool_func = lambda qr: courtroom5_search_tool(qr, txt)  # noqa: E731
    co_func = lambda qr: async_search_tool(qr, txt)  # noqa: E731

    return Tool(name=name,
                func=tool_func,
                coroutine=co_func,
                description=prompt)


def dynamic_courtroom5_tool_creator(t: SearchTool) -> Tool:
    """Create a custom courtroom5 search api tool for agents to use.

    Parameters
    ----------
    t : SearchTool
        The SearchTool object which describes the tool

    Returns
    -------
    Tool
        The tool created to be used by agents

    """
    name = t.name
    prompt = t.prompt
    txt = t.prefix

    async def async_search_tool(qr: str, txt: str) -> str:
        return dynamic_courtroom5_search_tool(qr, txt)

    tool_func = lambda qr: dynamic_courtroom5_search_tool(qr, txt)  # noqa: E731
    co_func = lambda qr: async_search_tool(qr, txt)  # noqa: E731

    return Tool(name=name,
                func=tool_func,
                coroutine=co_func,
                description=prompt)



def serpapi_tool_creator(t: SearchTool) -> Tool:
    """Create a serpapi tool for agents to use.

    Parameters
    ----------
    t : SearchTool
        The SearchTool object which describes the tool

    Returns
    -------
    Tool
        The tool created to be used by agents

    """
    name = t.name
    prompt = t.prompt
    txt = t.prefix

    async def async_search_tool(qr: str, txt: str) -> dict:
        return serpapi_tool(qr, txt)

    tool_func = lambda qr: serpapi_tool(qr, txt)  # noqa: E731
    co_func = lambda qr: async_search_tool(qr, txt)  # noqa: E731
    return Tool(name=name,
                func=tool_func,
                coroutine=co_func,
                description=prompt)


def openai_tool(t: SearchTool) -> dict:
    """Create a tool for openai agents to use.

    Parameters
    ----------
    t : SearchTool
        The SearchTool object which describes the tool

    Returns
    -------
    dict
        The description of tool created to be used by agents

    """
    body = {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.prompt,
            "parameters": {
                "type": "object",
                "properties": {
                    "qr": {
                        "type": "string",
                        "description": "The search text",
                    },
                },
                "required": ["qr"],
            },
        },
    }
    if t.method == SearchMethodEnum.courtlistener:
        # arg definitions
        body["function"]["parameters"]["properties"].update(courtlistener_tool_args)
        # modify query text for semantic + keyword queries
        body["function"]["parameters"]["properties"]["qr"]["description"] = (
            "A semantic query to search for general concepts and terms."
        )
        # default tool definition
        if not t.prompt:
            body["function"]["description"] = FILTERED_CASELAW_PROMPT
    return body

def anthropic_tool(t: SearchTool) -> dict:
    """Create a tool for anthropic agents to use.

    Parameters
    ----------
    t : SearchTool
        The SearchTool object which describes the tool

    Returns
    -------
    dict
        The description of tool created to be used by agents

    """
    body = {
        "name": t.name,
        "description": t.prompt,
        "input_schema": {
            "type": "object",
            "properties": {
                "qr": {
                    "type": "string",
                    "description": "The search text",
                },
            },
            "required": ["qr"],
        },
    }
    if t.method == SearchMethodEnum.courtlistener:
        # add courtlistener arg definitions
        body["input_schema"]["properties"].update(courtlistener_tool_args)
        # modify query text for semantic + keyword queries
        body["input_schema"]["properties"]["qr"]["description"] = (
            "A semantic query to search for general concepts and terms."
        )
        # default tool definition
        if not t.prompt:
            body["description"] = FILTERED_CASELAW_PROMPT
    return body


def run_search_tool(tool: SearchTool, function_args: dict) -> str:
    """Create a search tool for an openai agent.

    Parameters
    ----------
    tool : SearchTool
        The SearchTool object which describes the tool
    function_args : dict
        The arguments to pass to the function

    Returns
    -------
    str
        The response from the search tool

    """
    function_response = None
    prf = tool.prefix
    qr = function_args["qr"]
    match tool.method:
        case SearchMethodEnum.serpapi:
            function_response = serpapi_tool(qr, prf)
        case SearchMethodEnum.dynamic_serpapi:
            function_response = dynamic_serpapi_tool(qr, prf)
        case SearchMethodEnum.google:
            function_response = google_search_tool(qr, prf)
        case SearchMethodEnum.courtlistener:
            tool_jurisdiction = None
            tool_kw_query = None
            tool_after_date = None
            tool_before_date = None
            if "jurisdiction" in function_args:
                tool_jurisdiction = function_args["jurisdiction"].lower()
            if "keyword-qr" in function_args:
                tool_kw_query = function_args["keyword-qr"]
            if "after-date" in function_args:
                tool_after_date = function_args["after-date"]
            if "before-date" in function_args:
                tool_before_date = function_args["before-date"]
            request = OpinionSearchRequest(
                query=qr,
                k=3,
                jurisdictions=tool_jurisdiction,
                keyword_query=tool_kw_query,
                after_date=tool_after_date,
                before_date=tool_before_date,
            )
            function_response = courtlistener_search(request)
        case SearchMethodEnum.courtroom5:
            function_response = courtroom5_search_tool(qr, prf)
        case SearchMethodEnum.dynamic_courtroom5:
            function_response = dynamic_courtroom5_search_tool(qr, prf)
    return str(function_response)


def search_toolset_creator(bot: BotRequest) -> list:
    """Create a search toolset for the bot from all the search tools.

    Parameters
    ----------
    bot : BotRequest
        Bot object

    Returns
    -------
    list
        The list of search tools formatted for the bot engine

    """
    toolset = []
    for t in bot.search_tools:
        match bot.chat_model.engine:
            case EngineEnum.openai:
                toolset.append(openai_tool(t))
            case EngineEnum.anthropic:
                toolset.append(anthropic_tool(t))
    return toolset
