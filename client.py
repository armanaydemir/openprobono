import gradio as gr
import random
import time
import os
import json

from langchain.vectorstores import Vectara
from langchain.vectorstores.vectara import VectaraRetriever
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ConversationChain

from typing import Dict

from langchain import PromptTemplate, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        payload = {
            "inputs": [
                [
                 {"role": "user", "content": prompt}],
                ],
                "parameters": {"max_new_tokens": 1000, "top_p": 0.6, "temperature": 0.1},
        }

        input_str = json.dumps(
            payload,
        )
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generation"]["content"]

with gr.Blocks(title="OpenProBono",
    #font=gr.themes.GoogleFont("Open Sans"),
    css="footer {visibility: hidden}") as demo:

    content_handler = ContentHandler()

    sage = SagemakerEndpoint(
            endpoint_name="jumpstart-dft-meta-textgeneration-llama-2-7b-f",
            region_name="us-east-1",
            model_kwargs={"temperature": 1e-10},
            endpoint_kwargs={"CustomAttributes": "accept_eula=true"},
            content_handler=content_handler,
        )

    llm = ChatOpenAI(temperature=1.0, model='gpt-3.5-turbo-0613')
    
    def add_text(history, text):
        history = history + [(text, None)]
        return history, gr.update(value="", interactive=False)


    def add_file(history, file):
        loader = TextLoader(file.name)
        file_text = loader.load()

        history = history + [((file.name,), None)]
        return history


    #actually generates the text and uses langchain (this is where handoff between frontend and backend is)
    def bot(history, context):
        print(context)
        print("context^^")
        PROMPT = ""
        if context != "":
            PROMPT += "Pay attention and remember information below, which will help to answer the question or imperative after the context ends.\n"
            PROMPT += context
            PROMPT += "\nReference the information in the document sources provided within the context above.\n"
        PROMPT += "The following is a conversation between a human and an AI. The AI is a helpful assistant. If the AI does not know the answer to a question, it truthfully says it does not know.\n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:"
        PROMPT_TEMPLATE = PromptTemplate(input_variables=['history', 'input'], output_parser=None, partial_variables={}, template=PROMPT, template_format='f-string', validate_template=True)

        history_langchain_format = ChatMessageHistory()
        for i in range(0, len(history)-1):
            (human, ai) = history[i]
            history_langchain_format.add_user_message(human)
            history_langchain_format.add_ai_message(ai)
        memory = ConversationBufferMemory(return_messages=True, chat_memory=history_langchain_format)
        conversation = ConversationChain(
            llm=sage,
            memory=memory,
            prompt=PROMPT_TEMPLATE,
        )

        bot_message = conversation.run(history[-1][0])
        history[-1][1] = bot_message #.split("AI: ")[1]
        yield history

    def openai_bot(history, context):
        PROMPT = ""
        if context != "":
            PROMPT += "Pay attention and remember information below, which will help to answer the question or imperative after the context ends.\n"
            PROMPT += context
            PROMPT += "\nReference the information in the document sources provided within the context above.\n"
        PROMPT += "The following is a conversation between a human and an AI. The AI is a helpful assistant. If the AI does not know the answer to a question, it truthfully says it does not know.\n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:"
        PROMPT_TEMPLATE = PromptTemplate(input_variables=['history', 'input'], output_parser=None, partial_variables={}, template=PROMPT, template_format='f-string', validate_template=True)

        history_langchain_format = ChatMessageHistory()
        for i in range(0, len(history)-1):
            (human, ai) = history[i]
            history_langchain_format.add_user_message(human)
            history_langchain_format.add_ai_message(ai)
        openai_memory = ConversationBufferMemory(return_messages=True, chat_memory=history_langchain_format)
        openai_conversation = ConversationChain(
            llm=llm,
            memory=openai_memory,
            prompt=PROMPT_TEMPLATE,
        )

        bot_message = openai_conversation.run(history[-1][0]) #llm(history_langchain_format).content
        # print(openai_memory)
        history[-1][1] = bot_message #.split("AI: ")[1]
        yield history
    


    gr.Markdown("OpenProBono")
    with gr.Row():
        sagebot = gr.Chatbot(
            [],
            elem_id="sagemaker-llama",
            label="llama2-7b",
            #bubble_full_width=True,
            #avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
        )
        openai = gr.Chatbot(
            [],
            elem_id="gpt3.5-turbo",
            label="gpt3.5",
            #bubble_full_width=True,
            #avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
        )

    with gr.Row():
        contxt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter any context you want the AI to reference", #, or upload an image",
            container=False,
        )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter query", #, or upload an image",
            container=False,
        )
        subbtn = gr.Button("Submit")
        #btn = gr.UploadButton("📁", file_types=["text"])

    with gr.Row():
        clearsage = gr.ClearButton([txt, sagebot])
        clearopenai = gr.ClearButton([txt, openai])


    txt_msg = txt.submit(add_text, [sagebot, txt], [sagebot, txt], queue=False).then(
        bot, [sagebot, contxt], sagebot
    )
    txt_msg = txt.submit(add_text, [openai, txt], [openai, txt], queue=False).then(
        openai_bot, [openai, contxt], openai
    )

    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    sub_msg = subbtn.click(add_text, [sagebot, txt], [sagebot, txt], queue=False).then(
        bot, [sagebot, contxt], sagebot
    )
    sub_msg = subbtn.click(add_text, [openai, txt], [openai, txt], queue=False).then(
       openai_bot, [openai, contxt], openai
    )

    sub_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    
    # file_msg = btn.upload(add_file, [openai, btn], [openai], queue=False).then(
    #     bot, openai, openai
    # )

    gr.Markdown("This demo is a beta meant for informational purposes, demonstrating the abilities of our current technology. Data sent in demo is not guaranteed to be kept private. We will keep iterating on this demo, so keep an eye out for frequent updates.")

demo.queue()

demo.launch(root_path="/",favicon_path="./missing.ico")#auth=("sage", "kiwi"))
