from anyio.from_thread import start_blocking_portal
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import gradio as gr
import langchain
from langchain import PromptTemplate
from langchain.agents import AgentExecutor, AgentOutputParser, AgentType, LLMSingleActionAgent, initialize_agent, Tool, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, UnstructuredURLLoader
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.prompts import BaseChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AgentAction, AgentFinish, AIMessage, HumanMessage
from multiprocessing import Pool
import os
from queue import Queue
from typing import Any, List, Union
from serpapi import GoogleSearch
import sys
import uuid

# two main components: chat, bot
# - "___chat" is the actualy chat history / output on the screen
# - "___bot" is the function which calls the llm endpoint with some prompts and context and langchain magic

#makes it so it logs everything made by langchain basically
langchain.debug = True

#manually set the api key for now
GoogleSearch.SERP_API_KEY = "e6e9a37144cdd3e3e40634f60ef69c1ea6e330dfa0d0cde58991aa2552fff980"

#setting up firebase
cred = credentials.Certificate("../../creds.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

#setting up root path for firebase db purposes
if(len(sys.argv) < 3):
    root_path = "gradio_"
elif(sys.argv[1] == "/"):
    root_path = ""
else:
    root_path = sys.argv[1][1:] + "_"

##----------------------- frontend -----------------------##

#script for google analytics
ga_script = """
async () => {
    const script = document.createElement("script");
    script.onload = () =>  console.log("tag manager loaded") ;
    script.src = "https://www.googletagmanager.com/gtag/js?id=G-1FSYB9S6X6";
    document.head.appendChild(script);

    const script2 = document.createElement("script");
    script2.innerHTML = `
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-1FSYB9S6X6');
    `;
    document.head.appendChild(script2);
}
"""

#script for email submission event google analytics
email_ga_script = """
(email) => {
    gtag('event', 'email_submission', {
      'email': email,
    })
    return email
}
"""

#script for chat submission event google analytics
chat_ga_script = """
(chat) => {
    console.log(chat)
    console.log("chat loaded") 
    gtag('event', 'chat_submission', {
      'chat_length': chat.length,
    })
    return chat
}
"""

example_prompts = {
    "Criminal Law": [
        "What are my rights if I'm arrested?",
        "Is it legal to record a conversation without consent?",
        "What is the difference between assault and battery?",
        "Can you explain the process of plea bargaining?"
    ],
    "Family Law": [
        "How is child custody determined during a divorce?",
        "What are the legal requirements for getting a restraining order?",
        "What's the process for adoption in my state?",
        "How does spousal support work?"
    ],
    "Employment Law": [
        "Can my employer fire me without cause?",
        "What should I do if I'm facing workplace discrimination?",
        "What are the wage and hour laws in my jurisdiction?",
        "How do I negotiate a fair employment contract?"
    ],
    "Intellectual Property Law": [
        "How do I copyright my creative work?",
        "What is the process for filing a patent?",
        "What constitutes fair use in copyright law?",
        "How can I protect my company's trademarks?"
    ],
    "Real Estate Law": [
        "What's the process for buying a home and closing a deal?",
        "Can you explain zoning laws and their impact on property use?",
        "What are my rights and responsibilities as a tenant?",
        "How do easements work in real estate?"
    ],
    "Personal Injury Law": [
        "What should I do if I've been injured in a car accident?",
        "How can I prove liability in a personal injury case?",
        "What damages can I claim in a personal injury lawsuit?",
        "What is the statute of limitations for personal injury claims?"
    ],
    "Immigration Law": [
        "What are the different types of U.S. visas available?",
        "How does the naturalization process work for permanent residents?",
        "Can you explain the asylum application process?",
        "What are the consequences of overstaying a visa?"
    ]
}

def toggle_examples(state):
    state = not state
    #if examples are to be shown
    if(state):
        button_text = "Back"
    #if examples are to be hidden
    else:
        button_text = "Example Prompts"
    return gr.update(value=button_text), gr.update(visible = not state), gr.update(visible = not state), gr.update(visible = not state), gr.update(visible = state), state

def hide_examples(state):
    #if examples are currently shown, change state
    if(state):
        state = not state
    button_text = "Example Prompts"
    return gr.update(value=button_text), gr.update(visible = not state), gr.update(visible = not state), gr.update(visible = not state), gr.update(visible = state), state

def get_uuid_id():
    return str(uuid.uuid4())

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

    session = gr.State(get_uuid_id)

    def add_text(history, text):
        history = history + [(text, None)]
        return history, gr.update(value="", interactive=False)

    gr.Markdown("OpenProBono")
    with gr.Row() as chat_row:
        openai_chat = gr.Chatbot(
            [],
            elem_id="chat",
            label="OpenProBono",
            show_label=True,
        )

    with gr.Row() as input_row:
        txt = gr.Textbox(
            scale=4,
            label="input",
            show_label=False,
            placeholder="Enter query",
            container=False,
        )
        subbtn = gr.Button("Submit", variant="primary")
    
    examples_shown = gr.State(False)
    example_prompts_button = gr.Button("Example Prompts")

    with gr.Accordion("Details", open=False) as details_accordion:
        with gr.Row() as tool_row:
            t1txt = gr.Textbox(
                value="site:*.gov | site:*.edu",
                scale=4,
                label="Enter list of whitelisted urls for search with google syntax",
                show_label=True,
                container=True,
                interactive=True,
            )
            t1prompt = t1txt = gr.Textbox(
                value="Useful for when you need to answer questions or find resources about government and laws. Always cite your sources.",
                scale=4,
                label="Enter prompt for search",
                show_label=True,
                container=True,
                interactive=True,
            )
        with gr.Row() as tool_row:
            t2txt = gr.Textbox(
                value="site:*case.law",
                scale=4,
                label="Enter list of whitelisted urls for search with google syntax",
                show_label=True,
                container=True,
                interactive=True,
            )
            t2prompt = gr.Textbox(
                value="Use for finding case law. Always cite your sources.",
                scale=4,
                label="Enter prompt for search",
                show_label=True,
                container=True,
                interactive=True,
            )
        with gr.Row() as user_prompt_row:
            user_prompt = gr.Textbox(
                value="",
                scale=4,
                label="Enter additional system prompt",
                show_label=True,
                container=True,
                interactive=True,
            )
        gr.Markdown("This demo is a beta meant for informational purposes, demonstrating the abilities of our current technology and to compare different variations of models, prompting methods, document upload, and other features as we continually improve. The data sent in the demo is not guaranteed to be kept private. We will keep iterating on this demo, so keep an eye out for frequent updates. This is not legal advice. Learn more at www.openprobono.com.")
        clearopenai = gr.ClearButton([txt, openai_chat])

    with gr.Row() as email_row:    
        emailtxt = gr.Textbox(
            scale=4,
            label="input",
            show_label=False,
            placeholder="Enter your email to sign up for updates",
            container=False,
            type="email",
        )
        emailbtn = gr.Button("Submit")

    with gr.Column(visible=False) as examples_box:
        for prompt in example_prompts:
            with gr.Accordion(prompt, open=False):
                for example in example_prompts[prompt]:
                    exbtn = gr.Button(example)
                    exbtn.click(lambda x: x, exbtn, txt, queue=False).then(toggle_examples, [examples_shown], [example_prompts_button, chat_row, details_accordion, email_row, examples_box, examples_shown], queue=False)
    
    #connecting frontend interactions to backend
    example_prompts_button.click(toggle_examples, [examples_shown], [example_prompts_button, chat_row, details_accordion, email_row, examples_box, examples_shown], queue=False)

    ##----------------------- backend   (llm stuff)-----------------------##
    class MyCallbackHandler(BaseCallbackHandler):
        def __init__(self, q):
            self.q = q
        def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            self.q.put(token)

    def openai_bot(history, t1txt, t1prompt, t2txt, t2prompt, user_prompt, session):
        print('debug1')
        q = Queue()
        job_done = object()

        history_langchain_format = ChatMessageHistory()
        for i in range(0, len(history)-1):
            (human, ai) = history[i]
            history_langchain_format.add_user_message(human)
            history_langchain_format.add_ai_message(ai)
        memory = ConversationBufferMemory(return_messages=True, chat_memory=history_langchain_format, memory_key="memory")
        print('debug2')
        ##----------------------- tools -----------------------##
        def gov_search(q):
            data = {"search": t1txt + " " + q, 'prompt':t1prompt,'timestamp': firestore.SERVER_TIMESTAMP}
            db.collection(root_path + "search").document(session).collection('searches').document("search" + get_uuid_id()).set(data)
            return process_search(GoogleSearch({
                'q': t1txt + " " + q,
                'num': 5
                }).get_dict())

        def case_search(q):
            data = {"search": t2txt + " " + q, 'prompt': t2prompt, 'timestamp': firestore.SERVER_TIMESTAMP}
            db.collection(root_path + "search").document(session).collection('searches').document("search" + get_uuid_id()).set(data)
            return process_search(GoogleSearch({
                'q': t2txt + " " + q,
                'num': 5
                }).get_dict())

        async def async_gov_search(q):
            return gov_search(q)

        async def async_case_search(q):
            return case_search(q)

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
                pool = Pool(5)
                new_dict['organic_results'] = pool.map(search_helper_summarizer, results['organic_results'])

            return new_dict


        #Definition and descriptions of tools aviailable to the bot
        tools = [
            Tool(
                name="government-search",
                func=gov_search,
                coroutine=async_gov_search,
                description=t1prompt,
            ),
            Tool(
                name="case-search",
                func=case_search,
                coroutine=async_case_search,
                description=t2prompt,
            )
        ]
        tool_names = [tool.name for tool in tools]
        print('debug3')
        ##----------------------- end of tools -----------------------##
        #------- agent definition -------#
        # Set up the base template
        template = user_prompt + """Complete the user's request as best you can. You have access to the following tools:

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
        print('debug4')
        async def task(prompt):
            print('debug5')
            #definition of llm used for bot
            bot_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-0613', request_timeout=60*5)#, streaming=True, callbacks=[MyCallbackHandler(q)])
            agent_kwargs = {
                "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
            }
            print('debug6')
            llm_chain = LLMChain(llm=bot_llm, prompt=prompt)
            agent = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:"],
                allowed_tools=tool_names
            )
            print('debug7')
            agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory, verbose=True)
            print('debug8')
            ret = await agent_executor.arun(prompt)
            q.put(job_done)
            return ret

        with start_blocking_portal() as portal:
            portal.start_task_soon(task, history[-1][0])

            content = ""
            while True:
                next_token = q.get(True)
                if next_token is job_done:
                    break
                content += next_token
                history[-1][1] = content

                yield history
        

    ##----------------------- end of backend  (llm stuff)-----------------------##

    #storing conversations and emails in firebase
    def store_conversation(conversation, t1txt, t1prompt, t2txt, t2prompt, user_prompt, session):
        (human, ai) = conversation[-1]
        data = {"human": human, "ai": ai, 't1txt': t1txt, "t1prompt":t1prompt, "t2txt":t2txt, "t2prompt":t2prompt, 'user_prompt': user_prompt, 'timestamp':  firestore.SERVER_TIMESTAMP}
        db.collection(root_path + "conversations").document(session).collection('conversations').document("msg" + str(len(conversation))).set(data)

    def store_email(email, session):
        doc_ref = db.collection(root_path + "emails").document(session).set({"email": email, 'timestamp': firestore.SERVER_TIMESTAMP})
        print(email)
        print("^^ this is the email ^^")

    #corresponds to enter in the text box
    txt_msg = txt.submit(lambda: gr.update(interactive=False), None, [txt], queue=False).then(
        add_text, [openai_chat, txt], [openai_chat, txt], queue=False
    ).then(
        hide_examples, [examples_shown], [example_prompts_button, chat_row, details_accordion, email_row, examples_box, examples_shown], queue=False
    ).then(
        lambda x: x, [openai_chat], openai_chat, _js=chat_ga_script
    ).then(
        openai_bot, [openai_chat, t1txt, t1prompt, t2txt, t2prompt, user_prompt, session], [openai_chat]
    ).then(
        lambda: gr.update(interactive=True), None, [txt], queue=False
    ).then(store_conversation, [openai_chat, t1txt, t1prompt, t2txt, t2prompt, user_prompt, session], None, queue=False)

    #corresponds to clicking the submit button
    sub_msg = subbtn.click(lambda: gr.update(interactive=False), None, [txt], queue=False).then(
        add_text, [openai_chat, txt], [openai_chat, txt], queue=False, api_name="submit"
    ).then(
        hide_examples, [examples_shown], [example_prompts_button, chat_row, details_accordion, email_row, examples_box, examples_shown], queue=False
    ).then(
        lambda x: x, [openai_chat], openai_chat, _js=chat_ga_script
    ).then(
        openai_bot, [openai_chat, t1txt, t1prompt, t2txt, t2prompt, user_prompt, session], [openai_chat]
    ).then(
        lambda: gr.update(interactive=True), None, [txt], queue=False
    ).then(
        store_conversation, [openai_chat, t1txt, t1prompt, t2txt, t2prompt, user_prompt, session], None, queue=False
    )

    #hitting enter and clicking submit for email
    email_txt = emailtxt.submit(lambda x: x, [emailtxt], [emailtxt], queue=False, _js=email_ga_script).then(
        store_email, [emailtxt, session], None, queue=False
    )
    email_msg = emailbtn.click(store_email, [emailtxt, session], None, queue=False, _js=email_ga_script).then(
        store_email, [emailtxt, session], None, queue=False
    )

    #loading google analytics script
    app.load(None, None, None, _js=ga_script)
##----------------------- frontend -----------------------##
    
app.queue()

#using command line arguments to set port and root path
if(len(sys.argv) < 3):
    app.launch(share=True, favicon_path="./missing.ico")
else:
    app.launch(root_path=sys.argv[1], server_port=int(sys.argv[2]), favicon_path="./missing.ico")
