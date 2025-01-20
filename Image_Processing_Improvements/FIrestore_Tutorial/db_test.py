import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("Image_Processing_Improvements\FIrestore_Tutorial\serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()
print("Firestore connection successful!")
