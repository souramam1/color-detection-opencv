import firebase_admin
from firebase_admin import credentials, firestore

# VERY IMPORTANT - THE PATH MUST BE THE RELATIVE PATH OR IT CANNOT FIND THE SDK FILE
cred = credentials.Certificate("Image_Processing_Improvements\Firestore_Tutorial\serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()
db.collection('outputDevices').document('Camera').set(
    {
        'status' : False
    }
)
