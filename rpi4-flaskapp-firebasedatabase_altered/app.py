from flask import Flask, render_template, request
import firebase_admin
from firebase_admin import credentials, db
import datetime

app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("C:\Users\MaiaRamambason\OneDrive - Imperial College London\Desktop\Year5\UCL_BIP\Object_Detection_Trial\color-detection-opencv\rpi4-flaskapp-firebasedatabase_altered\hybrid-phys-firebase-adminsdk-vu6l4-f7ad67ad03.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://hybrid-phys-default-rtdb.europe-west1.firebasedatabase.app'
})

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        username = request.form['username']
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Push data to Firebase Realtime Database
        user_data_ref = db.reference('user_data')
        new_entry = user_data_ref.push({
            'username': username,
            'timestamp': current_time
        })

    # Retrieve data from Firebase Realtime Database
    user_data = db.reference('user_data').get()

    return render_template('index.html', user_data=user_data)

if __name__ == '__main__':
    app.run(debug=True)
