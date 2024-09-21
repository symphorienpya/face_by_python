import cv2
import gridfs
from pymongo import MongoClient

# MongoDB Connection
client = MongoClient('mongodb://localhost:27017/')
db = client['face_database']
fs = gridfs.GridFS(db)

# Load OpenCV's pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def save_face_to_mongodb(face_roi):
    """
    Saves the detected face image to MongoDB.
    """
    _, encoded_image = cv2.imencode('.jpeg', face_roi)
    binary_face_image = encoded_image.tobytes()
    fs.put(binary_face_image, filename="face_image.jpeg")
    print("Thank you !")

def capture_face():
    """
    Captures video from the webcam and allows saving the face on key press.
    """
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Capture Face', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):  # Press 'A' to capture and store the face
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face_roi = frame[y:y+h, x:x+w]
                    save_face_to_mongodb(face_roi)  # Save the first detected face
                    break
            else:
                print("No face detected to capture.")
        elif key == ord('p'):  # Press 'p' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_face()