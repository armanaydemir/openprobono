import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate("../../creds.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


#getting conversations and showing most recent ones

print(db.collection('conversations').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(10).get())