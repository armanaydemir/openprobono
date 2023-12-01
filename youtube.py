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

# Two Karpathy lecture videos
urls = ["https://youtu.be/kCc8FmEb1nY", "https://youtu.be/VMj-3S1tku0"]

# Directory to save audio files
save_dir = "./audio/"

audio = YoutubeAudioLoader(urls, save_dir)
print('debug1')
# Transcribe the videos to text
loader = GenericLoader(audio, OpenAIWhisperParser())
print('debug2')
docs = loader.load()
print('debug3')

# Combine doc
combined_docs = [doc.page_content for doc in docs]
text = " ".join(combined_docs)
print('debug4')

# Split them
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
splits = text_splitter.split_text(text)
print('debug5')

# Build an index
embeddings = OpenAIEmbeddings()
vectordb = FAISS.from_texts(splits, embeddings)

# Build a QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
)

# Ask a question!
query = "Why do we need to zero out the gradient before backprop at each step?"
print(qa_chain.run(query))
