import numpy as np
import cv2
from tensorflow.keras.models import load_model
import json

# Load the trained model
model = load_model("poker_card_classifier.h5")

# Load class indices
with open("class_indices.json", "r") as json_file:
    class_indices = json.load(json_file)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Preprocess the frame
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize

    # Make prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    # Get label from class indices
    label = list(class_indices.keys())[predicted_class]
    text = f"{label}: {confidence * 100:.2f}%"

    # Display the prediction on the frame
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Card Classification", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
