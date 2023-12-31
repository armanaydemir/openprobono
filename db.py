import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate("../../creds.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


#getting conversations and showing most recent ones
past = datetime.now() - timedelta(days=4)

docs = db.collection('conversationsClone').limit(10).get()
for doc in docs:
    # print(f'{doc.id} => {doc.to_dict()}')
    subdocs = db.collection('conversationsClone').document(doc.id).collection('conversations').get()
    # print(subdocs)
    for msg in subdocs:
        # print(f'{msg.id} => {msg.to_dict()}')
        if(msg.to_dict()['timestamp'] > past):
            print(f'{msg.id} => {msg.to_dict()}')
            
    print("------------------")