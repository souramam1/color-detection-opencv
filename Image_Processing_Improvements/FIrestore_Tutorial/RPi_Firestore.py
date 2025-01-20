import firebase_admin
from firebase_admin import credentials, firestore

# Set the path to the service account key file using correct path format
# Use double backslashes or forward slashes to avoid escape sequence issues


# VERY IMPORTANT - The path must be correct and accessible
cred = credentials.Certificate("/home/pi/Documents/color-detection-opencv/Image_Processing_Improvements/FIrestore_Tutorial/serviceAccountKey.json")  # Forward slashes

# Initialize Firebase app with the credentials
firebase_admin.initialize_app(cred)

# Access Firestore
db = firestore.client()

print("Firestore connection successful!")

# Write data to Firestore
db.collection('outputDevices').document('Camera').set({
    'status': False
})

db.collection('outputDevices').document('Screen').set({
    'status': False
})
db.collection('outputDevices').document('Test_3').set({
    'status': False
})
