import firebase_admin
from firebase_admin import credentials

cred = credentials.Certificate("path/to/serviceAccountKey.json")
firebase_admin.initialize_app(cred)


# # Firestore operations
# try:
#     db = firestore.client()

#     # Writing data to Firestore
#     db.collection('outputDevices').document('piCamera').set({
#         'status': False
#     })

#     db.collection('inputDevices').document('keyboardV').set({
#         'value': False
#     })

#     print("Data written to Firestore successfully.")
# except Exception as e:
#     print(f"Error writing to Firestore: {e}")
