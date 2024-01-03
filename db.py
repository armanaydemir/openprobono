import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import gradio as gr
import pytz

cred = credentials.Certificate("../../creds.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def loadChat(collection_name, chat_id):
    docs = db.collection(collection_name).document(chat_id).collection('conversations').get()
    history = []
    for doc in docs:
        human = doc.to_dict()['human']
        ai = doc.to_dict()['ai']
        history.append((human, ai))
    return gr.Chatbot(
        history,
        show_label=False,
    )

def getConvosInLast(num_days, collection_name):
    #getting conversations and showing most recent ones
    past = datetime.datetime.now() - datetime.timedelta(days=num_days)
    past = past.replace(tzinfo=pytz.UTC)
    docs = db.collection(collection_name).limit(1000).get()
    convos_to_return = []
    for doc in docs:
        subdocs = db.collection(collection_name).document(doc.id).collection('conversations').get()
        for msg in subdocs:
            # print(f'{msg.id} => {msg.to_dict()}')
            time = msg.to_dict()['timestamp']
            # print(time)
            if(time > past):
                doc.timestamp = time
                doc.num_msg = len(subdocs)
                convos_to_return.append(doc)
                break
    convos_to_return.sort(key=lambda x: x.timestamp, reverse=True)
    return convos_to_return

def createListFromConvos(num_days, collection_name, chat):
    convos = getConvosInLast(num_days, collection_name)
    text = ''
    with gr.Column() as convo_list:
        for convo in convos:
            text += "ID: " + convo.id + " Time: " + str(convo.timestamp) + " #MSG: " + str(convo.num_msg) + "\n"
    return text
    

with gr.Blocks(title="OPB DB", analytics_enabled=False) as app:
    with gr.Column() as conversations:
        num_days = gr.Number(value=1, label="Number of days to look back")
        collection_name = gr.Textbox(value="conversationsCloneJan3", label="Collection name")
        refresh_conversations = gr.Button(value="Get Conversations")
        conversation_textbot = gr.Textbox(value="List of Conversations", interactive="False")

    with gr.Column() as chat:
        chat_id = gr.Textbox(value="id", label="Chat ID")
        fetch_chat = gr.Button(value="Load A Chat")
        chat = gr.Chatbot()

    refresh_conversations.click(createListFromConvos, inputs=[num_days, collection_name, chat], outputs=[conversation_textbot])
    fetch_chat.click(loadChat, inputs=[collection_name, chat_id], outputs=[chat])
       
app.launch(share=True)