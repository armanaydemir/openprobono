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
import gradio as gr

template = """Respond in the same style as the context below.
{context}
Question: {question}
Response:"""
rag_prompt_custom = PromptTemplate.from_template(template)

def process(url, query):
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

    # Build a QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        chain_type="stuff",
        prompt=rag_prompt_custom,
        retriever=vectordb.as_retriever(),
    )

    return qa_chain.run(query)

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

    with gr.Row() as output_row:
        output = gr.Textbox(
            scale=40,
            label="input",
            show_label=False,
            placeholder="AI response",
            container=False,
        )

    subbtn.click(process, [url_txt, txt], outputs=[output])
    

app.queue()
app.launch(share=True)
