from flask import Flask, render_template, request, redirect, url_for, Response, session, flash
import cv2
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
from ultralytics import YOLO
from pymongo import MongoClient
import bcrypt
import os
import logging
import requests

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client['user_database']
users_collection = db['users']

# Load Violence Detection Model
violence_model_path = r'C:\Users\91600\OneDrive\Desktop\Real Time  Threat Detection System\my_flask_app\Suraksha-dhi-Real_Time_Threat_Detection_System-\violence_detection\model\modelnew (1).h5'
try:
    violence_model = load_model(violence_model_path)
    violence_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])
    app.logger.info("Violence detection model loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading violence model: {e}")
    violence_model = None

# Load Weapon Detection Model
weapon_model_path = r'C:\Users\91600\OneDrive\Desktop\Real Time  Threat Detection System\my_flask_app\Suraksha-dhi-Real_Time_Threat_Detection_System-\weapon_detection\best.pt'
try:
    weapon_model = YOLO(weapon_model_path)
    app.logger.info("Weapon detection model loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading weapon model: {e}")
    weapon_model = None


# Telegram bot settings
TELEGRAM_BOT_TOKEN = '7782595464:AAHGzlNG7EGojO94FEjwKha2bCKtXtlz3ok'
TELEGRAM_CHAT_ID = '5421262809'  # Replace with the target user's chat ID

# Function to send Telegram notifications
import requests  # Add this line at the top of your script

# Function to send Telegram notification
def send_telegram_message(message):
    try:
        # Replace with your own Telegram Bot API token and chat ID
        bot_token = '7782595464:AAHGzlNG7EGojO94FEjwKha2bCKtXtlz3ok'
        chat_id = '5421262809'
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {'chat_id': chat_id, 'text': message}
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            app.logger.error(f"Failed to send notification. Response: {response.text}")
        else:
            app.logger.info("Notification sent successfully.")
    except Exception as e:
        app.logger.error(f"Error sending notification: {e}")


# Function to check allowed file types for video uploads
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess Violence Detection Frame
def preprocess_violence_frame(frame):
    frame = cv2.resize(frame, (128, 128))
    frame = frame / 255.0
    return frame

# Display Results on Frame
def display_combined_results(frame, violence_detected, weapon_boxes): 
    # Display violence detection status
    text = "Violence Detected" if violence_detected else "No Violence Detected"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if violence_detected else (0, 255, 0), 2)

    # Display weapon detections with confidence scores
    for detection in weapon_boxes:
        # Ensure the format is correct: [x1, y1, x2, y2, confidence]
        x1, y1, x2, y2 = map(int, detection[:4])  # Convert coordinates to int
        conf = float(detection[4])  # Keep confidence as float
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"Weapon: {conf:.2f}"  # Display confidence as float
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame


# Combined Detection Stream Function
def detection_stream(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        app.logger.error("Unable to open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            app.logger.error("No frame received from source.")
            break

        # Violence Detection
        violence_frame = preprocess_violence_frame(frame)
        violence_preds = violence_model.predict(np.expand_dims(violence_frame, axis=0))[0] if violence_model else np.zeros(2)
        violence_detected = (np.mean(violence_preds) > 0.7)

        # Weapon Detection
        weapon_preds = weapon_model(frame) if weapon_model else []
        weapon_boxes = [[*box.xyxy[0], box.conf[0]] for box in weapon_preds[0].boxes] if weapon_preds else []

        # Send Telegram notifications if violence or weapon is detected
        if violence_detected:
            send_telegram_message("Violence detected!")
        if weapon_boxes:
            send_telegram_message(f"Weapon detected with confidence: {weapon_boxes[0][4]:.2f}")


        # Display Combined Results on Frame
        output_frame = display_combined_results(frame, violence_detected, weapon_boxes)

        # Encode the frame for streaming
        _, buffer = cv2.imencode('.jpg', output_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for the login page
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users_collection.find_one({"username": username})
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            session['username'] = username
            app.logger.info(f"{username} logged in successfully.")
            return redirect(url_for('choose_feed'))
        else:
            flash("Invalid credentials.", "error")
            app.logger.warning(f"Invalid login attempt for user: {username}")
    return render_template('login.html')

# Route for the sign-up page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if users_collection.find_one({"username": username}):
            flash("Username already exists. Please choose a different one.", "error")
            return redirect(url_for('signup'))
        else:
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            users_collection.insert_one({"username": username, "password": hashed_password.decode('utf-8')})
            flash("Account created successfully! Please log in.", "success")
            app.logger.info(f"New account created for user: {username}")
            return redirect(url_for('login'))
    return render_template('signup.html')

# Ensure the user is logged in before accessing feed options
@app.route('/choose_feed')
def choose_feed():
    if 'username' not in session:
        flash("Please log in to access this page.", "warning")
        return redirect(url_for('login'))
    return render_template('choose_feed.html')

# Route for the feed configuration page (for RTSP URL)
@app.route('/feed', methods=['GET', 'POST'])
def feed():
    if 'username' not in session:
        flash("Please log in to access this page.", "warning")
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        rtsp_url = request.form['rtsp_url']
        session['rtsp_url'] = rtsp_url  # Store RTSP URL in session
        return redirect(url_for('video_feed_cctv'))  # Redirect to CCTV feed
    return render_template('feed.html')

# Route for video upload
@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if 'username' not in session:
        flash("Please log in to upload videos.", "warning")
        return redirect(url_for('login'))

    if request.method == 'POST':
        video_file = request.files.get('video_file')
        print(f"Received video file: {video_file}")  # Debugging
        if video_file and allowed_file(video_file.filename):
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            file_path = os.path.join('uploads', video_file.filename)
            print(f"Saving file to: {file_path}")  # Debugging
            try:
                video_file.save(file_path)
                session['uploaded_video'] = file_path
                app.logger.info(f"Uploaded video saved to: {file_path}")
                return redirect(url_for('video_feed_uploaded'))
            except Exception as e:
                app.logger.error(f"Error saving uploaded video: {e}")
                flash("Error saving video. Please try again.", "error")
        else:
            print("File not allowed")  # Debugging
            flash("Invalid file type. Please upload a valid video file.", "error")
    return render_template('upload_video.html')


# Route for video feed from uploaded video
@app.route('/video_feed_uploaded')
def video_feed_uploaded():
    video_path = session.get('uploaded_video', '')
    if not video_path:
        flash("No video file uploaded.", "error")
        return redirect(url_for('upload_video'))
    return Response(uploaded_video_stream(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

# Function to capture and process uploaded video for weapon detection
def uploaded_video_stream(video_path):
    if video_path is None or not os.path.exists(video_path):
        app.logger.error("No valid uploaded video found.")
        return b''  # No valid uploaded video found
    
    cap = cv2.VideoCapture(video_path)  # Open the uploaded video
    if not cap.isOpened():
        app.logger.error(f"Unable to open uploaded video at {video_path}")
        return b''  # Unable to open the video file
    
    while True:
        ret, frame = cap.read()
        if not ret:
            app.logger.info("End of video reached or no frame received.")
            break  # End of the video file

        # Weapon Detection
        weapon_preds = weapon_model(frame) if weapon_model else []
        weapon_boxes = [[*box.xyxy[0], box.conf[0]] for box in weapon_preds[0].boxes] if weapon_preds else []

        if weapon_boxes:
            send_telegram_message(f"Weapon detected with confidence: {weapon_boxes[0][4]:.2f}")



        # Display Combined Results on Frame
        output_frame = display_combined_results(frame, violence_detected=False, weapon_boxes=weapon_boxes)

        # Encode the frame for streaming
        _, buffer = cv2.imencode('.jpg', output_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

# Video feed route for live camera feed
@app.route('/video_feed')
def video_feed():
    return Response(detection_stream(0), mimetype='multipart/x-mixed-replace; boundary=frame')

# Video feed route for RTSP stream
@app.route('/video_feed_cctv')
def video_feed_cctv():
    rtsp_url = session.get('rtsp_url', '')
    if not rtsp_url:
        flash("RTSP URL not provided.", "error")
        return redirect(url_for('choose_feed'))
    return Response(detection_stream(rtsp_url), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("You have been logged out.", "success")
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)
