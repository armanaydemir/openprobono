import gradio as gr
import langchain
from langchain import PromptTemplate
from langchain.agents import AgentExecutor, AgentType, LLMSingleActionAgent, initialize_agent, Tool, ZeroShotAgent
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, UnstructuredURLLoader
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.prompts import BaseChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage
from multiprocessing import Pool
import os
from serpapi import GoogleSearch
import sys
from typing import List, Union

# two main components: chat, bot
# - "___chat" is the actualy chat history / output on the screen
# - "___bot" is the function which calls the llm endpoint with some prompts and context and langchain magic

#makes it so it logs everything made by langchain basically
langchain.debug = True

#manually set the api key for now
GoogleSearch.SERP_API_KEY = "e6e9a37144cdd3e3e40634f60ef69c1ea6e330dfa0d0cde58991aa2552fff980"

##----------------------- tools -----------------------##

#General Search (no filters)
def general_search(q):
    return process_search(GoogleSearch({
        'q': q,
        'num': 5
        }).get_dict())

#Government Search (filtered on whitelist sites of relialbe sources for government))
def gov_search(q):
    return process_search(GoogleSearch({
        'q': "site:*.gov | site:*scholar.google.com | site:*case.law | site:*findlaw.com " + q,
        'num': 5
        }).get_dict())

#Helper function for concurrent processing of search results, calls the summarizer llm
def search_helper_summarizer(result):
    result.pop("displayed_link", None)
    result.pop("favicon", None)
    result.pop("about_page_link", None)
    result.pop("about_page_serpapi_link", None)

    summary_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-16k-0613')
    llm_input = """Summarize this web page in less than 100 words.

    Web Page:
    """
    llm_input += str(UnstructuredURLLoader(urls=[result["link"]]).load())
    result["page_summary"] = summary_llm.predict(llm_input)
    return result

#Filter search results retured by serpapi to only include relavant results
def process_search(results):
    new_dict = {}
    # if('sports_results' in results):
    #     new_dict['sports_results'] = results['sports_results']
    if('organic_results' in results):
        new_dict['organic_results'] = results['organic_results']
        pool = Pool()
        pool.map(search_helper_summarizer, new_dict["organic_results"])

    return new_dict

#Definition and descriptions of tools aviailable to the bot
tools = [
    Tool(
        name="search",
        func=general_search,
        description="useful for when you need to answer questions about current events. You should ask targeted questions. Always cite your sources.",
    ),
    Tool(
        name="government-search",
        func=gov_search,
        description="useful for when you need to answer questions or find resources about government and laws. Always cite your sources.",
    )
]
##----------------------- end of tools -----------------------##

##----------------------- backend   (llm stuff)-----------------------##
#definition of llm used for bot
bot_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-0613')

#need a better way to store emails
def print_email(email):
    print(email)
    print("^^ this is the email ^^")
    return email

#------- agent definition -------#
# Set up the base template
template = """Complete the user's request as best you can. You have access to the following tools:

{tools}

The following is the chat history so far:
{memory}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question, including your sources

These were previous tasks you completed:



Begin!

{input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
    
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps", "memory"]
)

output_parser = CustomOutputParser()
#------- end of agent definition -------#

def openai_bot(history):
    history_langchain_format = ChatMessageHistory()
    for i in range(0, len(history)-1):
        (human, ai) = history[i]
        history_langchain_format.add_user_message(human)
        history_langchain_format.add_ai_message(ai)
    memory = ConversationBufferMemory(return_messages=True, chat_memory=history_langchain_format, memory_key="memory")

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    llm_chain = LLMChain(llm=gpt3_llm, prompt=prompt)
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory, verbose=True)
    bot_message = agent_executor.run(history[-1][0])

    bot_message = agent.run(history[-1][0])
    history[-1][1] = bot_message
    yield history
##----------------------- end of backend  (llm stuff)-----------------------##

##----------------------- frontend -----------------------##

#script for google analytics
ga_script = """
async () => {
    const script = document.createElement("script");
    script.onload = () =>  console.log("tag manager loaded") ;
    script.src = "https://www.googletagmanager.com/gtag/js?id=G-MKDNM9G2PQ";
    document.head.appendChild(script);

    const script2 = document.createElement("script");
    script2.onload = () => {
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-MKDNM9G2PQ');
    }
    document.head.appendChild(script2);
}
"""

with gr.Blocks(
    title="OpenProBono",
    theme=gr.themes.Default(
        primary_hue=gr.themes.colors.indigo, 
        secondary_hue=gr.themes.colors.blue,
        font=gr.themes.GoogleFont("Open Sans"),
        radius_size=gr.themes.sizes.radius_lg,
    ),
    css="footer {visibility: hidden}",
    analytics_enabled=False
    ) as app:
    
    def add_text(history, text):
        history = history + [(text, None)]
        return history, gr.update(value="", interactive=False)

    gr.Markdown("OpenProBono")
    with gr.Row():
        openai_chat = gr.Chatbot(
            [],
            elem_id="chat",
            label="OpenProBono",
            show_label=True,
        )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            label="input",
            show_label=False,
            placeholder="Enter query",
            container=False,
        )
        subbtn = gr.Button("Submit", variant="primary")
        clearopenai = gr.ClearButton([txt, openai_chat])
       
    with gr.Accordion("Details"):
        with gr.Row():
            emailtxt = gr.Textbox(
                scale=4,
                label="input",
                show_label=False,
                placeholder="Enter your email to sign up for updates",
                container=False,
            )
            emailbtn = gr.Button("Submit")
        gr.Markdown("This demo is a beta meant for informational purposes, demonstrating the abilities of our current technology and to compare different variations of models, prompting methods, document upload, and other features as we continually improve. The data sent in the demo is not guaranteed to be kept private. We will keep iterating on this demo, so keep an eye out for frequent updates. This is not legal advice. Learn more at www.openprobono.com.")

    #connecting frontend interactions to backend

    #corresponds to enter in the text box
    txt_msg = txt.submit(add_text, [openai_chat, txt], [openai_chat, txt], queue=False).then(
        openai_bot, [openai_chat], openai_chat
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    
    #corresponds to clicking the submit button
    sub_msg = subbtn.click(add_text, [openai_chat, txt], [openai_chat, txt], queue=False, api_name="submit").then(
        openai_bot, [openai_chat], openai_chat
    )
    sub_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    #hitting enter and clicking submit for email
    email_txt = emailtxt.submit(print_email, [emailtxt], [emailtxt], queue=False)
    email_msg = emailbtn.click(print_email, [emailtxt], [emailtxt], queue=False)

    #loading google analytics script
    app.load(None, None, None, _js=ga_script)
##----------------------- frontend -----------------------##
    
app.queue()

#using command line arguments to set port and root path
if(len(sys.argv) < 3):
    app.launch(share=True, favicon_path="./missing.ico")
else:
    app.launch(root_path=sys.argv[1], server_port=int(sys.argv[2]), favicon_path="./missing.ico")
