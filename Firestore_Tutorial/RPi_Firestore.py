import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
 
######## in the tutorial he puts stuff in here to control gpio pins #######

# OUTPUT DEVICE


# INPUT DEVICE # separates collections according to input and output

db = firestore.client()
db.collection('outputDevices').document('piCamera').set(
    {
        'status' : False
    }
)
db.collection('inputDevices').document('keyboardV').set(
    {
        'value' : False
    }
)


