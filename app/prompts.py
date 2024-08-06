"""Prompts to use with LLMs."""

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

MAX_NUM_TOOLS = 8

MULTIPLE_TOOLS_PROMPT = (
    "You are a legal expert, tasked with answering any question about law. ALWAYS use "
    "tools to answer questions.\n\n"
    "Combine tool results into a coherent answer. If you used a tool, ALWAYS return a "
    '"SOURCES" part in your answer.'
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You're a legal expert. Given a question and chunks of legal text, "
                "answer the user question. If none of the chunks answer the question, "
                "just say you don't know.\n\nHere are the chunks:{context}"
            ),
        ),
        ("human", "{question}"),
    ],
)

# https://docs.anthropic.com/claude/docs/tool-use-examples
# To prompt Sonnet or Haiku to better assess the user query before making tool calls,
# the following prompt can be used:"
ANTHROPIC_SONNET_TOOL_PROMPT = (
    "Answer the user's request using relevant tools (if they are available). "
    "Before calling a tool, do some analysis within <thinking></thinking> tags. "
    "First, think about which of the provided tools is the relevant tool to answer the "
    "user's request. Second, go through each of the required parameters of the "
    " relevant tool and determine if the user has directly provided or given enough "
    "information to infer a value. When deciding if the parameter can be inferred, "
    "carefully consider all the context to see if it supports a specific value. If all "
    "of the required parameters are present or can be reasonably inferred, close the "
    "thinking tag and proceed with the tool call. BUT, if one of the values for a "
    "required parameter is missing, DO NOT invoke the function (not even with fillers "
    "for the missing params) and instead, ask the user to provide the missing "
    "parameters. DO NOT ask for more information on optional parameters if it is not "
    "provided."
)

# based on moderation prompt from Anthropic's API:
# https://docs.anthropic.com/claude/docs/content-moderation
MODERATION_PROMPT = (
    "A human user is in dialogue with an AI. The human is asking the AI questions "
    "about legal information and resources. Here is the most recent request from "
    "the user:<user query>{user_input}</user query>\n\n"
    "If the user's request refers to harmful, pornographic, or illegal activities, "
    "reply with (Y). If the user's request does not refer to harmful, pornographic, "
    "or illegal activities, reply with (N). Reply with nothing else other than (Y) or "
    "(N)."
)

SUMMARY_PROMPT = (
    "Write a concise summary of the following text delimited by triple backquotes. "
    "Return your response in bullet points which covers the key points of the text."
    "\n\n```{text}```\n\n"
    "BULLET POINT SUMMARY:"
)

SUMMARY_MAP_PROMPT = (
    "Write a concise summary of the following text delimited by triple backquotes."
    "\n\n```{text}```\n\n"
    "CONCISE SUMMARY:"
)

SUMMARY_REFINE_PROMPT = (
    "Taking the following context delimited by triple backquotes into consideration:"
    "\n\n```{context}```\n\n"
    "Write a concise summary of the following text delimited by triple backquotes."
    "\n\n```{text}```\n\n"
    "CONCISE SUMMARY:"
)

HIVE_QA_PROMPT = (
    "You are a legal assistant that provides factual information about "
    "legal code based on the following prompts.  After using information "
    "from a selected prompt, please attempt to provide a reference to the "
    "section that the information was pulled from. If a question does not "
    "make sense or is not factually coherent, explain why instead of "
    "answering something not correct. Please respond clearly and concisely "
    "as if talking to the customer directly."
)


OPB_BOT_PROMPT = ChatPromptTemplate(
    input_variables=["agent_scratchpad", "input"],
    input_types={
        "chat_history": list,
        "agent_scratchpad": list,
    },
    messages=[
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(input_variables=[],template=MULTIPLE_TOOLS_PROMPT),
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(input_variables=["input"], template="{input}"),
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ],
)


#for bot.py
COMBINE_TOOL_OUTPUTS_TEMPLATE = """You are a legal expert, tasked with answering any questions about law. ALWAYS use tools to answer questions.

Combine tool results into a coherent answer. If you used a tool, ALWAYS return a "SOURCES" part in your answer.
"""

NEW_TEST_TEMPLATE = """You are a legal expert, tasked with answering any questions about law. ALWAYS use tools to answer questions.

You can use multiple tools and the same tool multiple times with different inputs with the goal of gettting relevant resulsts.

Considering the results from the tools you used, decide if another tool call could be useful in providing a comprehensive answer.

If not, then provide an answer to the user's question. ALWAYS return a "SOURCES" part in your answer.
"""


#these are not used currently
ANSWER_TEMPLATE = """GENERAL INSTRUCTIONS
    You are a legal expert. Your task is to compose a response to the user's question using the information in the given context.

    CONTEXT:
    {context}

    USER QUESTION
    {input}"""

TOOLS_TEMPLATE = """GENERAL INSTRUCTIONS
    You are a legal expert. Your task is to decide which tools to use to answer a user's question. You can use up to X tools, and you can use tools multiple times with different inputs as well.

    These are the tools which are at your disposal
    {tools}

    When choosing tools, use this template:
    {{"tool": "name of the tool", "input": "input given to the tool"}}

    USER QUESTION
    {input}

    ANSWER FORMAT
    {{"tools":["<FILL>"]}}"""

# for evaluations.py

EVALUATION_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback:"""

FILTERED_CASELAW_PROMPT = (
    "Use to find case law and optionally filter by jurisdiction and date range. "
    "You can search by semantic or keyword. For example, to search for "
    "cases related to workers compensation that cite the Jones Act, you "
    'can search semantically for "workers compensation" and search by '
    'keyword for "Jones Act". Be sure to enter dates in YYYY-MM-DD format. '
    "The citation for this tool should include a link in the following format: "
    "https://www.courtlistener.com/ + metadata['absolute_url']. "
)

VDB_PROMPT = (
    "This tool queries a database named {collection_name} "
    "and returns the top {k} results. "
    "The database description is: {description}."
)

COMPARISON_PROMPT = """You will be given a question, the true answer to that question, and two answers generated by AI language models in response to the question. Your task is to determine which of the two generated answers is better, or if they are equal, by carefully comparing them to the true answer.

Here is the question:
{question}

Here is the true answer to the question:
{true_answer}

Here is generated answer A:
{answer_a}

And here is generated answer B:
{answer_b}

First, analyze each generated answer and compare it to the true answer. Consider the following:

- How accurate is the generated answer compared to the true answer? Does it contain any incorrect statements or information?
- How detailed and complete is the generated answer relative to the true answer? Does it cover all the key points, or is important information missing? 
- How relevant and on-topic is the generated answer? Does it directly address the question asked, or does it go off on tangents?

Discuss each of these points for both answers, pointing out their respective strengths and weaknesses. Aim to be as objective as possible in your analysis.

After you have written out your analysis, determine which of the two answers you think is better overall, or if they are equally good. Remember, do not simply choose the longer answer - a shorter answer that efficiently covers the key points may be better than a longer one that includes extraneous or incorrect information. Focus on accuracy, relevance and completeness relative to the true answer.

The output format should look as follows: \"Feedback: {{Write out your analysis and reasoning}} [RESULT] {{A or B or Tie}}\"
Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output."""


OPINION_SUMMARY_BASE_PROMPT = """Write your summary as a list of bullet points with the following titles:

Parties
Introduction
Background
Procedural History
Issues Presented
Analysis
Holding

Provide exactly one bullet point per title. Each bullet point should be no longer than 3 sentences. Only use the titles given above, and do not change the titles."""

OPINION_SUMMARY_MAP_PROMPT = \
    "Your task is to summarize a judicial opinion. " +\
    OPINION_SUMMARY_BASE_PROMPT +\
    """ If there is not any information related to a title, write "I could not find any information." for that bullet point."""

OPINION_SUMMARY_REDUCE_PROMPT = \
    """Your task is to combine partial summaries of a judicial opinion into a single, comprehensive summary.""" +\
    OPINION_SUMMARY_BASE_PROMPT
