import langchain
from firebase_admin import firestore
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain, FlareChain, create_retrieval_chain
from langchain.document_loaders.youtube import YoutubeLoader
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import MessagesPlaceholder, PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.llms import OpenAI
from typing import Any
from anyio.from_thread import start_blocking_portal
from queue import Queue
from serpapi.google_search import GoogleSearch
import milvusdb

langchain.debug = True

# OPB bot main function
def opb_bot(
    history,
    bot_id,
    tools,
    user_prompt = "", 
    session = ""):

    class MyCallbackHandler(BaseCallbackHandler):
        def __init__(self, q):
            self.q = q
        def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            self.q.put(token)

    if(history[-1][0].strip() == ""):
        return "Hi, how can I assist you today?"
    else:
        q = Queue()
        job_done = object()

        bot_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-1106', request_timeout=60*5, streaming=True, callbacks=[MyCallbackHandler(q)])
        memory_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-1106')

        memory = ConversationSummaryBufferMemory(llm=memory_llm, max_token_limit=2000, memory_key="memory", return_messages=True)
        for i in range(1, len(history)-1):
            memory.save_context({'input': history[i][0]}, {'output': history[i][1]})

        ##----------------------- tools -----------------------##

        #Filter search results retured by serpapi to only include relavant results
        def filtered_search(results):
            new_dict = {}
            if('sports_results' in results):
                new_dict['sports_results'] = results['sports_results']
            if('organic_results' in results):
                new_dict['organic_results'] = results['organic_results']
            return new_dict

        toolset = []
        tool_names = []
        for t in tools:
            def search_tool(qr):
                data = {"search": t['txt'] + " " + qr, 'prompt': t['prompt'], 'timestamp': firestore.SERVER_TIMESTAMP}
                return filtered_search(GoogleSearch({
                    'q': t['txt'] + " " + qr,
                    'num': 5
                    }).get_dict())

            async def async_search_tool(qr):
                return search_tool(qr)

            toolset.append(Tool(
                name = t["name"],
                func = search_tool,
                coroutine = async_search_tool,
                description = t["prompt"]
            )) 
            tool_names.append(t["name"])

        ##----------------------- end of tools -----------------------##


        #------- agent definition -------#
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
                tools=toolset,
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
                    return content
                content += next_token
                
        


#TODO: cache vector db with bot_id
#TODO: do actual chat memory
#TODO: try cutting off intro and outro part of videos
def youtube_bot(
    history,
    bot_id,
    youtube_urls = [],
    user_prompt = "",
    session = ""):

    if(user_prompt is None or user_prompt == ""):
        user_prompt = "Respond in the same style as the youtuber in the context below."

    prompt_template = user_prompt + """
    \n\nContext: {context}
    \n\n\n\n
    Question: {question}
    Response:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}

    embeddings = OpenAIEmbeddings()
    bot_path = "./youtube_bots/" + bot_id
    try:
        vectordb = FAISS.load_local(bot_path, embeddings)
    except:
        text = ""
        for url in youtube_urls:
            try:
                # Load the audio
                loader = YoutubeLoader.from_youtube_url(
                    url, add_video_info=False
                )
                docs = loader.load()
                # Combine doc
                combined_docs = [doc.page_content for doc in docs]
                text += " ".join(combined_docs)
            except:
                print("Error occured while loading transcript from video with url: " + url)

        # Split them
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        splits = text_splitter.split_text(text)

        # Build an index
        vectordb = FAISS.from_texts(splits, embeddings)
        vectordb.save_local(bot_path)

    # Build a QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
    )

    query = history[-1][0]
    #check for empty query
    if(query.strip() == ""):
        return ""
    else:
        return qa_chain.run(query)
    
def db_query(database_name: str, query: str, k: int = 4, user: str = None):
    """
    Runs query on database_name and returns the top k chunks

    Args
        database_name: the name of a pymilvus.Collection
        query: the user query
        k: return the top k chunks
        user: the username for filtering user data

    Returns dict with success or failure message and a result if success
    """
    if milvusdb.check_params(database_name, query, k):
        return milvusdb.check_params(database_name, query, k)
    
    db = milvusdb.load_db(database_name)
    if user:
        retriever = milvusdb.FilteredRetriever(vectorstore=db.as_retriever(), user_filter=user, search_kwargs={"k": k})
    else:
        retriever = db.as_retriever(search_kwargs={"k": k})

    results = retriever.get_relevant_documents(query)
    results_json = [{"text": result.page_content,
                     "source": result.metadata["source"],
                     "page": result.metadata["page"]} for result in results]
    return {"message": "Success", "result": results_json}

def db_retrieve(database_name: str, query: str, k: int = 4, user: str = None):
    """
    Runs query on database_name and returns an answer along with the top k source chunks

    This should be similar to db_bot, but using newer langchain LCEL

    Args
        database_name: the name of a pymilvus.Collection
        query: the user query
        k: return the top k chunks
        user: the username for filtering user data

    Returns dict with success or failure message and a result if success
    """
    if milvusdb.check_params(database_name, query, k):
        return milvusdb.check_params(database_name, query, k)
    
    db = milvusdb.load_db(database_name)
    retrieval_qa_chat_prompt: ChatPromptTemplate = hub.pull("langchain-ai/retrieval-qa-chat")
    llm = OpenAI(temperature=0)
    if user:
        retriever = milvusdb.FilteredRetriever(vectorstore=db.as_retriever(), user_filter=user, search_kwargs={"k": k})
    else:
        retriever = db.as_retriever(search_kwargs={"k": k})
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    result = retrieval_chain.invoke({"input": query})
    cited_sources = []
    for doc in result["context"]:
        cited_sources.append({"source": doc.metadata["source"], "page": doc.metadata["page"]})
    return {"message": "Success", "result": {"answer": result["answer"].strip(), "sources": cited_sources}}

def db_bot(database_name: str, question: str, k: int = 4, user: str = None):
    """
    Runs the question query on database_name and returns an answer along with cited sources from the top k chunks

    Args
        database_name: the name of a pymilvus.Collection
        question: the user question
        k: return cited sources from the top k chunks
        user: the username for filtering user data

    Returns dict with success or failure message and a result if success
    """
    if milvusdb.check_params(database_name, question, k):
        return milvusdb.check_params(database_name, question, k)
    
    db = milvusdb.load_db(database_name)
    if user:
        retriever = milvusdb.FilteredRetriever(vectorstore=db.as_retriever(), user_filter=user, search_kwargs={"k": k})
    else:
        retriever = db.as_retriever(search_kwargs={"k": k})
    chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0),
                                                        chain_type="stuff",
                                                        retriever=retriever,
                                                        return_source_documents=True)
    result = chain.invoke({"question": question})
    answer = result["answer"]
    cited_sources = result["sources"].split(", ")
    source_docs = result["source_documents"]
    cited_sources_docs = []
    for cited_source in cited_sources:
        for doc in source_docs:
            if doc.metadata["source"] == cited_source:
                cited_sources_docs.append({"source": cited_source, "page": doc.metadata["page"]})
    return {"message": "Success", "result": {"answer": answer.strip(), "sources": cited_sources_docs}}

def db_flare(database_name: str, question: str, k: int = 4, user: str = None):
    """
    Runs the question query on database_name and returns an answer using a retriever that returns the top k chunks

    This uses Forward-Looking Active REtrieval augmented generation (FLARE) with langchain FlareChain

    Args
        database_name: the name of a pymilvus.Collection
        question: the user question
        k: return cited sources from the top k chunks
        user: the username for filtering user data

    Returns dict with success or failure message and a result if success
    """
    if milvusdb.check_params(database_name, question, k):
        return milvusdb.check_params(database_name, question, k)
    
    db = milvusdb.load_db(database_name)
    if user:
        retriever = milvusdb.FilteredRetriever(vectorstore=db.as_retriever(), user_filter=user, search_kwargs={"k": k})
    else:
        retriever = db.as_retriever(search_kwargs={"k": k})
    flare = FlareChain.from_llm(
        ChatOpenAI(temperature=0),
        retriever=retriever,
        max_generation_len=164,
        min_prob=0.4,
    )
    return {"message": "Success", "result": flare.invoke({"user_input": question})["response"].strip()}