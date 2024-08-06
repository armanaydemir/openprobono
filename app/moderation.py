from __future__ import annotations

from anthropic import Anthropic
from langfuse.decorators import observe
from openai import OpenAI

from app.models import AnthropicModelEnum, ChatModelParams, EngineEnum, OpenAIModelEnum
from app.prompts import MODERATION_PROMPT


def moderate(
    message: str,
    chatmodel: ChatModelParams | None = None,
    client: OpenAI | Anthropic | None = None,
) -> bool:
    """Moderate a message using the specified engine and model.

    Parameters
    ----------
    message : str
        The message to be moderated.
    chatmodel : ChatModelParams, optional
        The engine and model to use for moderation, by default
        openai and text-moderation-latest.
    client : OpenAI | anthropic.Anthropic, optional
        The client to use for the moderation request. If not specified,
        one will be created.

    Returns
    -------
    bool
        True if flagged; False otherwise.

    Raises
    ------
    ValueError
        If chatmodel.engine is not `openai` or `anthropic`.

    """
    if chatmodel is None:
        chatmodel = ChatModelParams(
            engine=EngineEnum.openai,
            model=OpenAIModelEnum.mod_latest,
        )
    match chatmodel.engine:
        case EngineEnum.openai:
            return moderate_openai(message, chatmodel.model, client)
        case EngineEnum.anthropic:
            return moderate_anthropic(message, chatmodel.model, client)
    msg = f"Unsupported engine: {chatmodel.engine}"
    raise ValueError(msg)


@observe()
def moderate_openai(
    message: str,
    model: str = OpenAIModelEnum.mod_latest,
    client: OpenAI | None = None,
) -> bool:
    """Moderate a message using OpenAI's Moderation API.

    Parameters
    ----------
    message : str
        The message to be moderated.
    model : str, optional
        The model to use for moderation, by default `text-moderation-latest`.
    client : OpenAI, optional
        The client to use, by default None.

    Returns
    -------
    bool
        True if flagged; False otherwise.

    """
    client = OpenAI() if client is None else client
    response = client.moderations.create(model=model, input=message)
    return response.results[0].flagged


@observe()
def moderate_anthropic(
    message: str,
    model: str = AnthropicModelEnum.claude_3_haiku,
    client: Anthropic | None = None,
) -> bool:
    """Moderate a message using an Anthropic model.

    Parameters
    ----------
    message : str
        The message to be moderated.
    model : str, optional
        The model to use for moderation, by default `claude-3-haiku-20240307`.
    client : Anthropic, optional
        The client to use, by default None.

    Returns
    -------
    bool
        True if flagged; False otherwise.

    """
    client = Anthropic() if client is None else client
    moderation_msg = {
        "role": "user",
        "content": MODERATION_PROMPT.format(user_input=message),
    }
    response = client.messages.create(
        model=model,
        max_tokens=10,
        temperature=0,
        messages=[moderation_msg],
    )
    return "Y" in response.content[-1].text.strip()
