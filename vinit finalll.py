import cv2
from deepface import DeepFace
import emoji

# Define the emotion labels
emotion_labels = {
    'angry': 'angry',
    'disgust': 'disgusted',
    'fear': 'fearful',
    'happy': 'happy',
    'sad': 'sad',
    'surprise': 'surprised',
    'neutral': 'neutral'
}

# Load the emoji icons
emoji_icons = {
    'angry': emoji.emojize(':angry_face:'),
    'disgusted': emoji.emojize(':nauseated_face:'),
    'fearful': emoji.emojize(':fearful_face:'),
    'happy': emoji.emojize(':smiling_face_with_smiling_eyes:'),
    'sad': emoji.emojize(':pensive_face:'),
    'surprised': emoji.emojize(':astonished_face:'),
    'neutral': emoji.emojize(':neutral_face:')
}

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if any face is detected
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Extract the face region
            face = frame[y:y + h, x:x + w]

            # Perform emotion detection
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)

            # Get the dominant emotion from the result list
            emotion = result[0]['dominant_emotion']

            # Print the emotion class with emoji
            emoji_icon = emoji_icons.get(emotion, '')
            print(f"Emotion: {emotion_labels[emotion]} {emoji_icon}")

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the emotion text near the face
            cv2.putText(frame, f"{emotion_labels[emotion]} {emoji_icon}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Webcam Feed', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
