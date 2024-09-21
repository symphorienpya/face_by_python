import cv2
import gridfs
import numpy as np
from pymongo import MongoClient

# MongoDB Connection
client = MongoClient('mongodb://localhost:27017/')
db = client['face_database']
fs = gridfs.GridFS(db)

# Load OpenCV's pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def retrieve_face_from_mongodb():
    """
    Retrieves the first stored face image from MongoDB.
    """
    file = fs.find_one({"filename": "face_image.jpeg"})
    if file:
        image_data = np.frombuffer(file.read(), np.uint8)
        stored_face = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        return stored_face
    return None

def save_face_to_mongodb(face_roi):
    """
    Saves the detected face image to MongoDB.
    """
    _, encoded_image = cv2.imencode('.jpeg', face_roi)
    binary_face_image = encoded_image.tobytes()
    fs.put(binary_face_image, filename="face_image.jpeg")
    print("thank you!")

def compare_faces(face_roi, stored_face):
    """
    Compare the detected face with the stored face.
    Returns True if they match, False otherwise.
    """
    # Resize the face_roi to match the stored_face dimensions
    face_roi_resized = cv2.resize(face_roi, (stored_face.shape[1], stored_face.shape[0]))

    # Calculate the difference
    difference = cv2.absdiff(face_roi_resized, stored_face)
    if np.mean(difference) < 30:  # Adjust threshold as necessary
        return True
    return False

def capture_and_match_face():
    """
    Captures video from the webcam and allows saving the face on key press,
    and checking for matches in real-time.
    """
    stored_face = retrieve_face_from_mongodb()
    if stored_face is None:
        print("No stored face found in MongoDB.")
        return

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        match_found = False

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            if compare_faces(face_roi, stored_face):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green for match
                match_found = True
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red for no match

        cv2.imshow('Face Matching', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Press 'c' to capture and store the face
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    save_face_to_mongodb(frame[y:y+h, x:x+w])  # Save the first detected face
                    break
            else:
                print("No face detected to capture.")
        elif key == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_match_face()
