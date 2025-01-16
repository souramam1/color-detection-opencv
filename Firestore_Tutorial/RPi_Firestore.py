import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
 
######## in the tutorial he puts stuff in here to control gpio pins #######

###########################################################################

db = firestore.client()
