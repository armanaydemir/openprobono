from langchain.document_loaders import YoutubeLoader
import langchain
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import (
    OpenAIWhisperParser
)
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.prompts import BaseChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AgentAction, AgentFinish, AIMessage, HumanMessage
from langchain.chains import ConversationalRetrievalChain
import gradio as gr

def process(history, url):
    print("history")
    print(history)
    loader = YoutubeLoader.from_youtube_url(
        url, add_video_info=False
    )

    docs = loader.load()

    # Combine doc
    combined_docs = [doc.page_content for doc in docs]
    text = " ".join(combined_docs)

    # Split them
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_text(text)

    # Build an index
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_texts(splits, embeddings)
    retriever = vectordb.as_retriever()

    # Build a memory
    history_txt = ""
    for i in range(0, len(history)-1):
        (human, ai) = history[i]
        history_txt += human + "\n" + ai + "\n"
    # history_langchain_format = ChatMessageHistory()
    # for i in range(0, len(history)-1):
    #     (human, ai) = history[i]
    #     history_langchain_format.add_user_message(human)
    #     history_langchain_format.add_ai_message(ai)
    # memory = ConversationBufferMemory(memory_key='memory', return_messages=True, output_key='answer')

    prompt_template = """Respond in the same style as the context below.
    {context}

    Chat History:
    """ + history_txt + """
    Question: {question}
    Response:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT} 

    # qa_chat = ConversationalRetrievalChain.from_llm(
    #     llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    #     memory=memory,
    #     retriever=retriever, 
    #     return_source_documents=False,
    #     combine_docs_chain_kwargs=chain_type_kwargs,
    # )

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
    )
    query = history[-1][0]
    history[-1][1] = qa_chain.run(query)
    return history

with gr.Blocks(
    title="Youtube QA",
    theme=gr.themes.Default(
        primary_hue=gr.themes.colors.indigo, 
        secondary_hue=gr.themes.colors.blue,
        font=gr.themes.GoogleFont("Open Sans"),
        radius_size=gr.themes.sizes.radius_lg,
    ),
    css="footer {visibility: hidden}",
    analytics_enabled=False
    ) as app:

    with gr.Row() as chat_row:
        chat = gr.Chatbot(
            [],
            elem_id="chat",
            label="Youtube QA",
            show_label=True,
        )

    def add_text(history, text):
        history = history + [(text, None)]
        return history

    with gr.Row() as url_row:
        url_txt = gr.Textbox(
            scale=4,
            label="input",
            show_label=False,
            placeholder="Enter youtube url",
            container=False,
            type="text",
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

    subbtn.click(lambda: gr.update(interactive=False), None, [txt], queue=False
    ).then(
        add_text, [chat, txt], chat
    ).then(
        process, [chat, url_txt], chat
    ).then(
        lambda: gr.update(interactive=True), None, [txt], queue=False
    )
    

app.queue()
app.launch(share=True)
