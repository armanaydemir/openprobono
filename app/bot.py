"""Defines the bot engines. The meaty stuff."""
import json

import openai
from anthropic import Anthropic
from anthropic.types.beta.tools import ToolsBetaContentBlock
from langfuse.decorators import langfuse_context, observe
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall

from app.chat_models import chat, chat_history
from app.models import BotRequest, ChatRequest
from app.moderation import moderate
from app.prompts import (
    MAX_NUM_TOOLS,
    MULTIPLE_TOOLS_PROMPT,
)
from app.search_tools import (
    run_search_tool,
    search_toolset_creator,
)
from app.vdb_tools import (
    run_vdb_tool,
    vdb_toolset_creator,
)

openai.log = "debug"

def stream_openai_response(response: ChatCompletion):
    full_delta_dict_collection = []
    no_yield = False
    for chunk in response:
        if(chunk.choices[0].delta.tool_calls or no_yield):
            no_yield = True
            full_delta_dict_collection.append(chunk.choices[0].delta.to_dict())
        else:
            chunk_content = chunk.choices[0].delta.content
            if(chunk_content):
                yield chunk.choices[0].delta.content

    tool_calls, current_dict = [], {}
    if(no_yield):
        current_dict = full_delta_dict_collection[0]
        for i in range(1, len(full_delta_dict_collection)):
            merge_dicts_stream_openai_completion(current_dict, full_delta_dict_collection[i])

        tool_calls = [ChatCompletionMessageToolCall.model_validate(tool_call)
                    for tool_call in current_dict["tool_calls"]]
    return tool_calls, current_dict


def merge_dicts_stream_openai_completion(dict1, dict2):
    for key in dict2:
        if key in dict1:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                merge_dicts_stream_openai_completion(dict1[key], dict2[key])
            elif isinstance(dict1[key], list) and isinstance(dict2[key], list):
                for item in dict2[key]:
                    item_index= item["index"]
                    del item["index"]
                    if(item_index < len(dict1[key])):
                       if("index" in dict1[key][item_index]):
                           del dict1[key][item_index]["index"]
                       merge_dicts_stream_openai_completion(dict1[key][item_index], item)
                    else:
                        dict1[key].append(item)
            else:
                dict1[key] += dict2[key]
        else:
            dict1[key] = dict2[key]


def openai_bot_stream(r: ChatRequest, bot: BotRequest):
    """Call bot using openai engine.

    Parameters
    ----------
    r : ChatRequest
        ChatRequest object, containing the conversation and session data
    bot : BotRequest
        BotRequest object, containing the bot data

    Returns
    -------
    str
        The response from the bot

    """
    if r.history[-1]["content"].strip() == "":
        return "Hi, how can I assist you today?"
    if moderate(r.history[-1]["content"].strip()):
        return (
            "I'm sorry, I can't help you with that. "
            "Please modify your message and try again."
        )

    client = OpenAI()
    messages = r.history
    messages.append({"role": "system", "content": bot.system_prompt})
    toolset = search_toolset_creator(bot) + vdb_toolset_creator(bot)
    kwargs = {
        "client": client,
        "tools": toolset,
        "tool_choice": "auto",  # auto is default, but we'll be explicit
        "temperature": 0,
        "stream": True,
    }

    # response is a ChatCompletion object
    response: ChatCompletion = chat(messages, bot.chat_model, **kwargs)
    tool_calls, current_dict = yield from stream_openai_response(response)
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        messages.append(current_dict)
        yield from openai_tools_stream(messages, tool_calls, bot, **kwargs)

def openai_tools_stream(
    messages: list[dict],
    tool_calls: list[ChatCompletionMessageToolCall],
    bot: BotRequest,
    **kwargs: dict,
):
    """Handle tool calls in the conversation for OpenAI engine.

    Parameters
    ----------
    messages : list[dict]
        The conversation messages
    tool_calls : list[ChatCompletionMessageToolCall]
        List of tool calls
    bot : BotRequest
        BotRequest object

    Returns
    -------
    messages : list[dict]
        The updated conversation messages with tool responses appended

    """
    tools_used = 0
    while tool_calls and tools_used < MAX_NUM_TOOLS:
        # TODO: run tool calls in parallel
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            vdb_tool = next(
                (t for t in bot.vdb_tools if function_name == t.name),
                None,
            )
            search_tool = next(
                (t for t in bot.search_tools if function_name == t.name),
                None,
            )
            # Step 3: call the function
            # Note: the JSON response may not always be valid;
            # be sure to handle errors
            yield f"\nRunning {function_name} tool with the following arguments: {function_args}"
            if vdb_tool:
                tool_response = run_vdb_tool(
                    vdb_tool,
                    function_args,
                )
            elif search_tool:
                tool_response = run_search_tool(
                    search_tool,
                    function_args,
                )
            else:
                tool_response = "error: unable to run tool"
            # Step 4: send the info for each function call and function response to
            # the model
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": tool_response,
                },
            )  # extend conversation with function response
            tools_used += 1
        # get a new response from the model where it can see the function response
        yield "\nAnalyzing tool results\n"
        response = chat(messages, bot.chat_model, **kwargs)
        tool_calls, current_dict = yield from stream_openai_response(response)
        messages.append(current_dict)


@observe(capture_input=False)
def openai_bot(r: ChatRequest, bot: BotRequest) -> str:
    """Call bot using openai engine.

    Parameters
    ----------
    r : ChatRequest
        ChatRequest object, containing the conversation and session data
    bot : BotRequest
        BotRequest object, containing the bot data

    Returns
    -------
    str
        The response from the bot

    """
    if r.history[-1]["content"].strip() == "":
        return "Hi, how can I assist you today?"
    if moderate(r.history[-1]["content"].strip()):
        return (
            "I'm sorry, I can't help you with that. "
            "Please modify your message and try again."
        )

    client = OpenAI()
    messages = r.history
    messages.append({"role": "system", "content": bot.system_prompt})
    toolset = search_toolset_creator(bot) + vdb_toolset_creator(bot)
    kwargs = {
        "client": client,
        "tools": toolset,
        "tool_choice": "auto",  # auto is default, but we'll be explicit
        "temperature": 0,
    }
    # tracing
    langfuse_context.update_current_observation(
        input=messages[-2]["content"], # input is last user message
    )
    langfuse_context.update_current_trace(
        session_id=r.session_id,
        metadata={"bot_id": r.bot_id},
    )

    # response is a ChatCompletion object
    response: ChatCompletion = chat(messages, bot.chat_model, **kwargs)
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        messages.append(response_message.model_dump(exclude={"function_call"}))
        response_message = openai_tools(messages, tool_calls, bot, **kwargs)
    return response_message.content

def openai_tools(
    messages: list[dict],
    tool_calls: list[ChatCompletionMessageToolCall],
    bot: BotRequest,
    **kwargs: dict,
):
    """Handle tool calls in the conversation for OpenAI engine.

    Parameters
    ----------
    messages : list[dict]
        The conversation messages
    tool_calls : list[ChatCompletionMessageToolCall]
        List of tool calls
    bot : BotRequest
        BotRequest object

    Returns
    -------
    ChatCompletionMessage
        The response message from the bot, should be final response.

    """
    tools_used = 0
    while tool_calls and tools_used < MAX_NUM_TOOLS:
        # TODO: run tool calls in parallel
        for tool_call in tool_calls:
            print("RUN TOOL")
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            vdb_tool = next(
                (t for t in bot.vdb_tools if function_name == t.name),
                None,
            )
            search_tool = next(
                (t for t in bot.search_tools if function_name == t.name),
                None,
            )
            # Step 3: call the function
            # Note: the JSON response may not always be valid;
            # be sure to handle errors
            if vdb_tool:
                tool_response = run_vdb_tool(vdb_tool, function_args)
            elif search_tool:
                tool_response = run_search_tool(search_tool, function_args)
            else:
                tool_response = "error: unable to run tool"
            # Step 4: send the info for each function call and function response to
            # the model
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": tool_response,
                },
            )  # extend conversation with function response
            tools_used += 1
        # get a new response from the model where it can see the function response
        response = chat(messages, bot.chat_model, **kwargs)
        response_message = response.choices[0].message
        messages.append(response_message)
        tool_calls = response_message.tool_calls
    return response_message

@observe(capture_input=False)
def anthropic_bot(r: ChatRequest, bot: BotRequest):
    if r.history[-1][0].strip() == "":
        return "Hi, how can I assist you today?"
    if moderate(r.history[-1][0].strip(), bot.chat_model):
        return (
            "I'm sorry, I can't help you with that. "
            "Please modify your message and try again."
        )
    messages = chat_history(r.history, bot.chat_model.engine)
    toolset = search_toolset_creator(bot) + vdb_toolset_creator(bot)
    client = Anthropic()
    # Step 1: send the conversation and available functions to the model
    kwargs = {
        "tools": toolset,
        "client": client,
        "system": MULTIPLE_TOOLS_PROMPT,
        "temperature": 0,
    }

    # Step 1.5: tracing
    # Anthropic system prompt does not go in messages list, so add it to the input
    langfuse_prompt_msg = [{"role": "system", "content": kwargs["system"]}]
    langfuse_context.update_current_observation(input={
        "input": messages[-1]["content"],
        "prompt": langfuse_prompt_msg,
    })
    langfuse_context.update_current_trace(
        session_id=r.session_id,
        metadata={"bot_id": r.bot_id},
    )

    response = chat(messages, bot.chat_model, **kwargs)
    messages.append({"role": response.role, "content": response.content})
    # Step 2: check if the model wanted to call a function
    tool_calls: list[ToolsBetaContentBlock] = [
        msg for msg in response.content if msg.type == "tool_use"
    ]
    return anthropic_tools(messages, tool_calls, bot, **kwargs)

def anthropic_tools(
    messages: list[dict],
    tool_calls: list[ToolsBetaContentBlock],
    bot: BotRequest,
    **kwargs: dict,
):
    tools_used = 0
    while tool_calls and tools_used < MAX_NUM_TOOLS:
        for tool_call in tool_calls:
            function_name = tool_call.name
            vdb_tool = next(
                (t for t in bot.vdb_tools if function_name == t.name),
                None,
            )
            search_tool = next(
                (t for t in bot.search_tools if function_name == t.name),
                None,
            )
            tool_response_msg = {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
            }
            # Step 3: call the function
            if vdb_tool:
                tool_response = run_vdb_tool(vdb_tool, tool_call.input)
            elif search_tool:
                tool_response = run_search_tool(search_tool, tool_call.input)
            else:
                tool_response = "error: unable to identify tool"
                tool_response_msg["is_error"] = True
            tool_response_msg["content"] = tool_response
            # extend conversation with function response
            messages.append(
                {
                    "role": "user",
                    "content": [tool_response_msg],
                },
            )
            tools_used += 1
        # Step 4: send info for each function call and function response to the model
        # get a new response from the model where it can see the function response
        response = chat(messages, bot.chat_model, **kwargs)
        messages.append({"role": response.role, "content": response.content})
        tool_calls = [msg for msg in response.content if msg.type == "tool_use"]
    content: list[ToolsBetaContentBlock] = messages[-1]["content"]
    return content[-1].text
