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
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, UnstructuredURLLoader
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.prompts import BaseChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AgentAction, AgentFinish, AIMessage, HumanMessage
from multiprocessing import Pool
import os
from queue import Queue
import re
from serpapi import GoogleSearch
import sys
from typing import Any, Dict, List, Optional, Union
import uuid

# two main components: chat, bot
# - "___chat" is the actualy chat history / output on the screen
# - "___bot" is the function which calls the llm endpoint with some prompts and context and langchain magic

#makes it so it logs everything made by langchain basically
langchain.debug = True

#manually set the api key for now
GoogleSearch.SERP_API_KEY = "5567e356a3e19133465bc68755a124268543a7dd0b2809d75b038797b43626ab"

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

#script for user-agent retreival (returns true if mobile)
#also makes it dark mode
user_agent_script = """
() => {
    document.body.classList.add('dark');

    let check = false;
    (function(a){if(/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(a)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(a.substr(0,4))) check = true;})(navigator.userAgent||navigator.vendor||window.opera);
    return check;
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

def get_uuid_id():
    return str(uuid.uuid4())


theme = gr.Theme.load("./new_theme.json")

default = gr.themes.Default(
        primary_hue=gr.themes.colors.indigo, 
        secondary_hue=gr.themes.colors.blue,
        font=gr.themes.GoogleFont("Open Sans"),
    )

with gr.Blocks(
    title="OpenProBono",
    theme=theme.set(
            body_background_fill="linear-gradient(to right, #1e244d, #183e1b)",
            body_background_fill_dark="linear-gradient(to right, #1e244d, #183e1b)"),
    css="""
    footer {visibility: hidden}
    .gradio-container {max-width: 100%!important; width: 100%!important; }
    #component-0 { height: 100%!important; min-height: 100%!important; max-height: 100%!important; overflow: scroll!important;}
    #chat_col {height: 95vh!important; min-height: 95vh!important; max-height: 95vh!important;}
    #tools_col_css {height: 90vh!important; overflow: scroll!important;}
    #therow {height: 95vh!important; min-height: 95vh!important; max-height: 95vh!important;}
    #chatbot {height: 100%!important; min-height: 100%!important; max-height: 100%!important; flex-grow: 5; overflow: scroll!important;}
    #chatrow { height: 70%!important; min-height: 70%!important; max-height: 70%!important; }
    #inputrow { height: 10%!important; min-height: 10%!important; max-height: 10%!important; }
    """,
    #chat_col {height: 90%!important; min-height: 90%!important; max-height: 90%!important;}
    # 
    # .contain { display: flex; flex-direction: column; }
    # component-0 { height: 100%; }
    # chatbot { flex-grow: 1; overflow: auto;}
    # """,
    analytics_enabled=False,
    ) as app:
    
    #loading user agent
    isMobile = gr.Checkbox(label="isMobile", visible=False)
    app.load(None, None, [isMobile], _js=user_agent_script)

    session = gr.State(get_uuid_id)
    examples_shown = gr.State(False)

    def toggle_examples(state):
        state = not state
        return gr.update(visible = not state), gr.update(visible = state), state

    def hide_examples(state):
        #if examples are currently shown, change state
        if(state):
            state = not state
        return gr.update(visible = not state), gr.update(visible = state), state

    def add_text(history, text):
        history = history + [[text, None]]
        return history, gr.update(value="", interactive=False)

    gr.Markdown("<a href=\"https://www.openprobono.com/\" target=\"_blank\" style=\"text-decoration:none!important;  color: white\">OpenProBono</a>")
    with gr.Row(elem_id="therow") as the_row:
        with gr.Column(scale=2, elem_id="chat_col") as chat_col:
            with gr.Row(elem_id="chatrow") as chat_row:
                openai_chat = gr.Chatbot(
                    [[None, "Hi! Ask me any legal questions you have!\n \nIf you don\'t know where to start, try looking at the example prompts!"]],
                    elem_id="chatbot",
                    label="OpenProBono",
                    show_label=False,
                )

            with gr.Row(elem_id="inputrow") as input_row:
                txt = gr.Textbox(
                    scale=10,
                    label="input",
                    show_label=False,
                    placeholder="Enter query",
                    container=False,
                )
                subbtn = gr.Button("Submit", variant="primary", scale=1, min_width=1)

            example_prompts_button = gr.Button("Example Prompts", visible=False)
        
        # with gr.Group() as tools_desktop_group:
        with gr.Column(scale=0, elem_id="tools_col_css") as tools_col:
            with gr.Tab("Examples"):
                for prompt in example_prompts:
                    with gr.Accordion(prompt, open=False):
                        for example in example_prompts[prompt]:
                            exbtn = gr.Button(example)
                            exbtn.click(lambda x: x, exbtn, txt, queue=False)

            admin_visible = "admin" in root_path or "staging" in root_path
            with gr.Tab("Tools", visible=admin_visible):
                with gr.Row() as tool_row:
                    t1name = gr.Textbox(
                        value="government-search",
                        scale=4,
                        label="Enter name for tool",
                        show_label=True,
                        container=True,
                        interactive=True,
                    )
                    t1txt = gr.Textbox(
                        value="site:*.gov | site:*.edu | site:*scholar.google.com",
                        scale=4,
                        label="Enter list of whitelisted urls for search with google syntax",
                        show_label=True,
                        container=True,
                        interactive=True,
                    )
                    t1prompt = gr.Textbox(
                        value="Useful for when you need to answer questions or find resources about government and laws. Always cite your sources.",
                        scale=4,
                        label="Enter prompt for search",
                        show_label=True,
                        container=True,
                        interactive=True,
                    )
                with gr.Row() as tool_row:
                    t2name = gr.Textbox(
                        value="case-search",
                        scale=4,
                        label="Enter name for tool",
                        show_label=True,
                        container=True,
                        interactive=True,
                    )
                    t2txt = gr.Textbox(
                        value="site:*case.law | site:*.gov | site:*.edu | site:*courtlistener.com | site:*scholar.google.com",
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

            with gr.Tab("Details"):
                gr.Markdown("OpenProBono AI is designed to assist users in finding relevant information and resources related to government and laws. While we strive to provide accurate and up-to-date information, it is important to note that the AI's results should be verified against official sources. The AI's findings should not be considered legal advice, and users should consult with legal professionals for specific legal matters. Additionally, the AI's recommendations and suggestions are based on algorithms and data analysis, and may not cover all possible scenarios or legal interpretations. The AI's developers and operators do not assume any liability for the accuracy, completeness, or reliability of the AI's results. Users are responsible for independently verifying the information and using their own judgment in making legal decisions. Learn more at www.openprobono.com.")

    with gr.Column(visible=False) as examples_box:
        back_button = gr.Button("Back")
        for prompt in example_prompts:
            with gr.Accordion(prompt, open=False):
                for example in example_prompts[prompt]:
                    exbtn = gr.Button(example)
                    exbtn.click(lambda x: x, exbtn, txt, queue=False).then(toggle_examples, [examples_shown], [the_row, examples_box, examples_shown], queue=False)
    
    #connecting frontend interactions to backend
    example_prompts_button.click(toggle_examples, [examples_shown], [the_row, examples_box, examples_shown], queue=False)
    back_button.click(toggle_examples, [examples_shown], [the_row, examples_box, examples_shown], queue=False)
    
    ##----------------------- backend   (llm stuff)-----------------------##
    class MyCallbackHandler(BaseCallbackHandler):
        def __init__(self, q):
            self.q = q
        def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            self.q.put(token)

    def openai_bot(history, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt, session):
        if(history[-1][0].strip() == ""):
            history[-1][1] = "Hi, how can I assist you today?"
            yield history 
        else:
            q = Queue()
            job_done = object()

            bot_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-0613', request_timeout=60*5, streaming=True, callbacks=[MyCallbackHandler(q)])
            memory_llm = OpenAI(temperature=0.0, model='gpt-3.5-turbo-0613')

            history_langchain_format = ChatMessageHistory()
            for i in range(1, len(history)-1):
                (human, ai) = history[i]
                history_langchain_format.add_user_message(human)
                history_langchain_format.add_ai_message(ai)
            memory = ConversationTokenBufferMemory(llm=memory_llm, max_token_limit=4000, return_messages=True, chat_memory=history_langchain_format, memory_key="memory")
            ##----------------------- tools -----------------------##

            def gov_search(q):
                data = {"search": t1txt + " " + q, 'prompt':t1prompt,'timestamp': firestore.SERVER_TIMESTAMP}
                db.collection(root_path + "search").document(session).collection('searches').document("search" + get_uuid_id()).set(data)
                return filtered_search(GoogleSearch({
                    'q': t1txt + " " + q,
                    'num': 5
                    }).get_dict())

            def case_search(q):
                data = {"search": t2txt + " " + q, 'prompt': t2prompt, 'timestamp': firestore.SERVER_TIMESTAMP}
                db.collection(root_path + "search").document(session).collection('searches').document("search" + get_uuid_id()).set(data)
                return filtered_search(GoogleSearch({
                    'q': t2txt + " " + q,
                    'num': 5
                    }).get_dict())

            async def async_gov_search(q):
                return gov_search(q)

            async def async_case_search(q):
                return case_search(q)

            #Filter search results retured by serpapi to only include relavant results
            def filtered_search(results):
                new_dict = {}
                if('sports_results' in results):
                    new_dict['sports_results'] = results['sports_results']
                if('organic_results' in results):
                    new_dict['organic_results'] = results['organic_results']
                return new_dict

            #Definition and descriptions of tools aviailable to the bot
            tools = [
                Tool(
                    name=t1name,
                    func=gov_search,
                    coroutine=async_gov_search,
                    description=t1prompt,
                ),
                Tool(
                    name=t1name,
                    func=case_search,
                    coroutine=async_case_search,
                    description=t2prompt,
                )
            ]
            ##----------------------- end of tools -----------------------##

            system_message = 'You are a helpful AI assistant. ALWAYS use tools to answer questions.'
            system_message += user_prompt
            system_message += '. If you used a tool, ALWAYS return a "SOURCES" part in your answer.'
            agent_kwargs = {
                "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
            }
            async def task(prompt):
                #definition of llm used for bot
                prompt = "Using the tools at your disposal, answer the following question: " + prompt
                agent = initialize_agent(
                    tools=tools,
                    llm=bot_llm,
                    agent=AgentType.OPENAI_FUNCTIONS,
                    verbose=False,
                    agent_kwargs=agent_kwargs,
                    memory=memory,
                    #return_intermediate_steps=True
                )
                agent.agent.prompt.messages[0].content = system_message
                ret = await agent.arun(prompt)
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
    def store_conversation(conversation, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt, session):
        (human, ai) = conversation[-1]
        data = {"human": human, "ai": ai, "t1name": t1name, 't1txt': t1txt, "t1prompt":t1prompt, "t2name": t2name, "t2txt":t2txt, "t2prompt":t2prompt, 'user_prompt': user_prompt, 'timestamp':  firestore.SERVER_TIMESTAMP}
        db.collection(root_path + "conversations").document(session).collection('conversations').document("msg" + str(len(conversation))).set(data)

    def store_email(email, session):
        doc_ref = db.collection(root_path + "emails").document(session).set({"email": email, 'timestamp': firestore.SERVER_TIMESTAMP})
        print(email)
        print("^^ this is the email ^^")

    #corresponds to enter in the text box
    txt_msg = txt.submit(lambda: gr.update(interactive=False), None, [txt], queue=False).then(
        add_text, [openai_chat, txt], [openai_chat, txt], queue=False
    ).then(
        hide_examples, [examples_shown], [example_prompts_button, the_row, examples_box, examples_shown], queue=False
    ).then(
        lambda x: x, [openai_chat], openai_chat, _js=chat_ga_script, queue=False
    ).then(
        openai_bot, [openai_chat, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt, session], [openai_chat]
    ).then(
        lambda: gr.update(interactive=True), None, [txt], queue=False
    ).then(
        store_conversation, [openai_chat, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt, session], None, queue=False
    )

    #corresponds to clicking the submit button
    sub_msg = subbtn.click(lambda: gr.update(interactive=False), None, [txt], queue=False).then(
        add_text, [openai_chat, txt], [openai_chat, txt], queue=False, api_name="submit"
    ).then(
        hide_examples, [examples_shown], [example_prompts_button, the_row, examples_box, examples_shown], queue=False
    ).then(
        lambda x: x, [openai_chat], openai_chat, _js=chat_ga_script, queue=False
    ).then(
        openai_bot, [openai_chat, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt, session], [openai_chat]
    ).then(
        lambda: gr.update(interactive=True), None, [txt], queue=False
    ).then(
        store_conversation, [openai_chat, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt, session], None, queue=False
    )

    #hitting enter and clicking submit for email
    # email_txt = emailtxt.submit(lambda x: x, [emailtxt], [emailtxt], queue=False, _js=email_ga_script).then(
    #     store_email, [emailtxt, session], None, queue=False
    # )
    # email_msg = emailbtn.click(store_email, [emailtxt, session], None, queue=False, _js=email_ga_script).then(
    #     store_email, [emailtxt, session], None, queue=False
    # )

    def isMobile_change(isMobile):
        return gr.update(visible=(not isMobile), render=(not isMobile), interactive=(not isMobile)), gr.update(visible=isMobile, render=isMobile, interactive=isMobile)
    isMobile.change(isMobile_change, [isMobile], [tools_col, example_prompts_button], queue=False)

    #loading google analytics script
    app.load(None, None, None, _js=ga_script)
##----------------------- frontend -----------------------##
    
app.queue()

#using command line arguments to set port and root path
if(len(sys.argv) < 3):
    app.launch(share=True, favicon_path="./missing.ico")
else:
    app.launch(root_path=sys.argv[1], server_port=int(sys.argv[2]), favicon_path="./missing.ico")