import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate("../../creds.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


#getting conversations and showing most recent ones

docs = db.collection('conversationsClone').limit(1000).get()
docs = [doc.data for doc in docs]
print(docs[0])