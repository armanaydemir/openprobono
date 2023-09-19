import gradio as gr
import random
import time
import os
import json
import sagemaker

from langchain.vectorstores import Vectara
from langchain.vectorstores.vectara import VectaraRetriever
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain, ConversationChain, LLMChain

from typing import Dict

from langchain import PromptTemplate, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler

#tparts of a model: chat, bot
# - chat is the actualy chat history / output on the screen
# - bot calls the llm endpoint with some prompts and context and langchain magic

from abc import abstractmethod
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.utils import enforce_stop_tokens
from langchain.llms.sagemaker_endpoint import SagemakerEndpoint
import boto3, time, os, uuid
from botocore.exceptions import ClientError

def wait_inference_file(output_url, failure_url, s3_client=None):
    s3_client = boto3.client("s3") if s3_client == None else s3_client
    bucket = output_url.split("/")[2]
    output_prefix = "/".join(output_url.split("/")[3:])
    failure_prefix = "/".join(failure_url.split("/")[3:])
    while True:
        try:
            response = s3_client.get_object(Bucket=bucket, Key=output_prefix)
            print(response)
            return response
        except ClientError as ex:
            if ex.response['Error']['Code'] == 'NoSuchKey':
                try:
                    response = s3_client.get_object(Bucket=bucket, Key=failure_prefix)
                    raise Exception(response['Body'].read().decode('utf-8'))
                except ClientError as ex:
                    if ex.response['Error']['Code'] == 'NoSuchKey':
                        # Wait for file to be generated
                        print("Waiting for file to be generated...")
                        time.sleep(5)
                        continue
                    else:
                        raise
            else:
                raise

        except Exception as e:
            print(e.__dict__)
            raise

class SagemakerAsyncEndpoint(SagemakerEndpoint):
    input_bucket: str = ""
    input_prefix: str = ""
    max_request_timeout: int = 90
    s3_client: Any
    sm_client: Any
    
    def __init__(self, input_bucket:str="", input_prefix:str="", max_request_timeout:int=90, **kwargs):
        """
        Initialize a Sagemaker asynchronous endpoint connector in Langchain
        Args:
            input_bucket: S3 bucket name where input files are stored.
            input_prefix: S3 prefix where input files are stored.
            max_request_timeout: Maximum timeout for the request in seconds - also used to validate if endpoint is in cold start
            kwargs: Keyword arguments to pass to the SagemakerEndpoint class.
        """
        super().__init__(**kwargs)
        region = self.region_name
        account = boto3.client("sts").get_caller_identity()["Account"]
        self.input_bucket = f'sagemaker-{region}-{account}' if input_bucket == "" else input_bucket
        self.input_prefix = f'async-endpoint-outputs/{self.endpoint_name}' if input_prefix == "" else input_prefix
        self.max_request_timeout = max_request_timeout
        self.s3_client = boto3.client("s3")
        self.sm_client = boto3.client("sagemaker")
        
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Sagemaker asynchronous inference endpoint.
        Args:
            prompt: The prompt to use for the inference.
            stop: The stop tokens to use for the inference.
            run_manager: The run manager to use for the inference.
            kwargs: Keyword arguments to pass to the SagemakerEndpoint class.
        Returns:
            The output from the Sagemaker asynchronous inference endpoint.
        """
        # Parse the SagemakerEndpoint class arguments
        _model_kwargs = self.model_kwargs or {}
        _model_kwargs = {**_model_kwargs, **kwargs}
        _endpoint_kwargs = self.endpoint_kwargs or {}
        
        # Transform the input to match SageMaker expectations
        body = self.content_handler.transform_input(prompt, _model_kwargs)
        content_type = self.content_handler.content_type
        accepts = self.content_handler.accepts

        # Verify if the endpoint is running
        response = self.sm_client.describe_endpoint(EndpointName=self.endpoint_name)
        endpoint_is_running = response["ProductionVariants"][0]["CurrentInstanceCount"] > 0

        # If the endpoint is not running, send an empty request to "wake up" the endpoint
        test_data = b""
        test_key = os.path.join(self.input_prefix, "test")
        self.s3_client.put_object(Body=test_data, Bucket=self.input_bucket, Key=test_key)
        if not endpoint_is_running:
            response = self.client.invoke_endpoint_async(
                EndpointName=self.endpoint_name,
                InputLocation="s3://{}/{}".format(self.input_bucket, test_key),
                ContentType=content_type,
                Accept=accepts,
                InvocationTimeoutSeconds=self.max_request_timeout, # timeout of 60 seconds to detect if it's not running yet
                **_endpoint_kwargs,
            )
            return "Error: Endpoint is not running - check back in ~10 minutes"
            raise Exception("Endpoint is not running - check back in ~10 minutes.")
        else:
            print("Endpoint is running! Proceeding to inference.")
        
        # Send request to the async endpoint
        request_key = os.path.join(self.input_prefix, f"request-{str(uuid.uuid4())}")
        self.s3_client.put_object(Body=body, Bucket=self.input_bucket, Key=request_key)
        response = self.client.invoke_endpoint_async(
            EndpointName=self.endpoint_name,
            InputLocation="s3://{}/{}".format(self.input_bucket, request_key),
            ContentType=content_type,
            Accept=accepts,
            InvocationTimeoutSeconds=self.max_request_timeout, # timeout 
            OffloadFolder="./",
            **_endpoint_kwargs,
        )
        print(response)
        print("THISIS RESPONSE")
        # Read the bytes of the file from S3 in output_url with Boto3
        output_url = response["OutputLocation"]
        failure_url = response["FailureLocation"]
        print(output_url)
        print(failure_url)
        response = wait_inference_file(output_url, failure_url, self.s3_client)
        text = self.content_handler.transform_output(response["Body"])
        if stop is not None:
            text = enforce_stop_tokens(text, stop)

        return text

class AsyncContentHandler(LLMContentHandler):
    content_type:str = "application/json"
    accepts:str = "application/json"
    len_prompt:int = 0

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        self.len_prompt = len(prompt)
        input_str = json.dumps({"inputs": prompt, "parameters": {"max_new_tokens": 100, "do_sample": False, "repetition_penalty": 1.1}})
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> str:
        response_json = output.read()
        res = json.loads(response_json)
        ans = res[0]['generated_text']
        return ans


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

    sage_llm = SagemakerEndpoint(
            endpoint_name="jumpstart-dft-meta-textgeneration-llama-2-7b-f",
            region_name="us-east-1",
            model_kwargs={"temperature": 1e-10},
            endpoint_kwargs={"CustomAttributes": "accept_eula=true"},
            content_handler=content_handler,
        )

    async_content_handler = AsyncContentHandler()

    async_endpoint = SagemakerAsyncEndpoint(
        endpoint_name="hf-text2text-flan-t5-xxl-2023-09-18-22-08-48-231",
        region_name=sagemaker.Session().boto_region_name,
        content_handler=async_content_handler,
    )

    # gpt3_llm = ChatOpenAI(temperature=1.0, model='gpt-3.5-turbo-0613')
    
    def add_text(history, text):
        history = history + [(text, None)]
        return history, gr.update(value="", interactive=False)


    def add_file(history, file):
        loader = TextLoader(file.name)
        file_text = loader.load()

        history = history + [((file.name,), None)]
        return history

    def async_bot(history, context):
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
            llm = async_endpoint,
            memory = memory,
            prompt = PROMPT_TEMPLATE,
        )
        

        bot_message = conversation.run(history[-1][0])
        history[-1][1] = bot_message #.split("AI: ")[1]
        yield history

    #actually generates the text and uses langchain (this is where handoff between frontend and backend is)
    def bot(history, context):
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
            llm = sage_llm,
            memory = memory,
            prompt = PROMPT_TEMPLATE,
        )

        bot_message = conversation.run(history[-1][0])
        history[-1][1] = bot_message #.split("AI: ")[1]
        yield history

    # def openai_bot(history, context):
    #     PROMPT = ""
    #     if context != "":
    #         PROMPT += "Pay attention and remember information below, which will help to answer the question or imperative after the context ends.\n"
    #         PROMPT += context
    #         PROMPT += "\nReference the information in the document sources provided within the context above.\n"
    #     PROMPT += "The following is a conversation between a human and an AI. The AI is a helpful assistant. If the AI does not know the answer to a question, it truthfully says it does not know.\n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:"
    #     PROMPT_TEMPLATE = PromptTemplate(input_variables=['history', 'input'], output_parser=None, partial_variables={}, template=PROMPT, template_format='f-string', validate_template=True)

    #     history_langchain_format = ChatMessageHistory()
    #     for i in range(0, len(history)-1):
    #         (human, ai) = history[i]
    #         history_langchain_format.add_user_message(human)
    #         history_langchain_format.add_ai_message(ai)
    #     openai_memory = ConversationBufferMemory(return_messages=True, chat_memory=history_langchain_format)
    #     openai_conversation = ConversationChain(
    #         llm=gpt3_llm,
    #         memory=openai_memory,
    #         prompt=PROMPT_TEMPLATE,
    #     )

    #     bot_message = openai_conversation.run(history[-1][0])
    #     history[-1][1] = bot_message #.split("AI: ")[1]
    #     yield history
    


    gr.Markdown("OpenProBono")
    with gr.Row():
        async_chat = gr.Chatbot(
            [],
            elem_id="flan-t5-xxl",
            label="flan-t5-xxl",
            #bubble_full_width=True,
            #avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
        )
        sage_chat = gr.Chatbot(
            [],
            elem_id="sagemaker-llama",
            label="llama2-7b",
            #bubble_full_width=True,
            #avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
        )
        # openai_chat = gr.Chatbot(
        #     [],
        #     elem_id="gpt3.5-turbo",
        #     label="gpt3.5",
        #     #bubble_full_width=True,
        #     #avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
        # )

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
        clearasync = gr.ClearButton([txt, async_chat])
        clearsage = gr.ClearButton([txt, sage_chat])
        # clearopenai = gr.ClearButton([txt, openai_chat])


    txt_msg = txt.submit(add_text, [sage_chat, txt], [sage_chat, txt], queue=False).then(
        bot, [sage_chat, contxt], sage_chat
    )
    txt_msg = txt.submit(add_text, [async_chat, txt], [async_chat, txt], queue=False).then(
        async_bot, [async_chat, contxt], async_chat
    )
    # txt_msg = txt.submit(add_text, [openai_chat, txt], [openai_chat, txt], queue=False).then(
    #     async_bot, [async_chat, contxt], openai_chat
    # )

    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    sub_msg = subbtn.click(add_text, [sage_chat, txt], [sage_chat, txt], queue=False).then(
        bot, [sage_chat, contxt], sage_chat
    )
    sub_msg = subbtn.click(add_text, [async_chat, txt], [async_chat, txt], queue=False).then(
        async_bot, [async_chat, contxt], async_chat
    )
    # sub_msg = subbtn.click(add_text, [openai_chat, txt], [openai_chat, txt], queue=False).then(
    #     async_bot, [async_chat, contxt], openai_chat
    # )

    sub_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    
    # file_msg = btn.upload(add_file, [openai, btn], [openai], queue=False).then(
    #     bot, openai, openai
    # )

    gr.Markdown("This demo is a beta meant for informational purposes, demonstrating the abilities of our current technology. Data sent in demo is not guaranteed to be kept private. We will keep iterating on this demo, so keep an eye out for frequent updates.")

demo.queue()

demo.launch(root_path="/",favicon_path="./missing.ico")
