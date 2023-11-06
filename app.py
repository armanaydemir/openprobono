import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import gradio as gr
import langchain
from langchain import PromptTemplate
from langchain.agents import AgentExecutor, AgentType, initialize_agent, Tool, ZeroShotAgent
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage
import os
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

##----------------------- tools -----------------------##

#General Search (no filters)
def general_search(q):
    return filtered_search(GoogleSearch({
        'q': q,
        'num': 5
        }).get_dict())

#Government Search (filtered on whitelist sites of relialbe sources for government))
def gov_search(q):
    return filtered_search(GoogleSearch({
        'q': "site:*.gov " + q,
        'num': 5
        }).get_dict())

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
bot_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-0613', request_timeout=60*5)

def openai_bot(history):
    history_langchain_format = ChatMessageHistory()
    for i in range(0, len(history)-1):
        (human, ai) = history[i]
        history_langchain_format.add_user_message(human)
        history_langchain_format.add_ai_message(ai)
    memory = ConversationBufferMemory(return_messages=True, chat_memory=history_langchain_format, memory_key="memory")

    system_message = 'You are a helpful AI assistant. '
    #system_message += user_prompt
    system_message += '. ALWAYS return a "SOURCES" part in your answer.'
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    agent = initialize_agent(
        tools=tools,
        llm=bot_llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )
    agent.agent.prompt.messages[0].content = system_message
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
    #if examples being shown
    if(state):
        button_text = "Back"
    else:
        button_text = "Example Prompts"
    return gr.update(value=button_text), gr.update(visible = not state), gr.update(visible = not state), gr.update(visible = not state), gr.update(visible = state), state

def hide_examples(state):
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
        # clearopenai = gr.ClearButton([txt, openai_chat])
    
    examples_shown = gr.State(False)
    example_prompts_button = gr.Button("Example Prompts")

    with gr.Accordion("Details", open=False) as details_accordion:
        gr.Markdown("This demo is a beta meant for informational purposes, demonstrating the abilities of our current technology and to compare different variations of models, prompting methods, document upload, and other features as we continually improve. The data sent in the demo is not guaranteed to be kept private. We will keep iterating on this demo, so keep an eye out for frequent updates. This is not legal advice. Learn more at www.openprobono.com.")
    
    with gr.Row() as email_row:    
        emailtxt = gr.Textbox(
            scale=4,
            label="input",
            show_label=False,
            placeholder="Enter your email to sign up for updates",
            container=False,
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

    def store_conversation(conversation, session):
        print(session)
        doc_ref = db.collection("conversations").document(session)
        new_convo = []
        for i in range(0, len(conversation)):
            (human, ai) = conversation[i]
            new_convo.append({"human": human, "ai": ai})
        doc_ref.set({"conversation": new_convo})

    def store_email(email, session):
        print(session)
        doc_ref = db.collection("emails").document(session).set({"email": email})
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
        openai_bot, [openai_chat], [openai_chat]
    ).then(
        lambda: gr.update(interactive=True), None, [txt], queue=False
    ).then(store_conversation, [openai_chat, session], None, queue=False)

    #corresponds to clicking the submit button
    sub_msg = subbtn.click(lambda: gr.update(interactive=False), None, [txt], queue=False).then(
        add_text, [openai_chat, txt], [openai_chat, txt], queue=False, api_name="submit"
    ).then(
        hide_examples, [examples_shown], [example_prompts_button, chat_row, details_accordion, email_row, examples_box, examples_shown], queue=False
    ).then(
        lambda x: x, [openai_chat], openai_chat, _js=chat_ga_script
    ).then(
        openai_bot, [openai_chat], [openai_chat]
    ).then(
        lambda: gr.update(interactive=True), None, [txt], queue=False
    ).then(store_conversation, [openai_chat, session], None, queue=False)

    #hitting enter and clicking submit for email
    email_txt = emailtxt.submit(store_email, [emailtxt, session], None, queue=False, _js=email_ga_script)
    email_msg = emailbtn.click(store_email, [emailtxt, session], None, queue=False, _js=email_ga_script)

    #loading google analytics script
    app.load(None, None, None, _js=ga_script)
##----------------------- frontend -----------------------##
    
app.queue()

#using command line arguments to set port and root path
if(len(sys.argv) < 3):
    app.launch(share=True, favicon_path="./missing.ico")
else:
    app.launch(root_path=sys.argv[1], server_port=int(sys.argv[2]), favicon_path="./missing.ico")
