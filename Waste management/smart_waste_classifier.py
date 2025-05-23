import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('waste_model_vgg19.h5')

# Class labels
class_labels = ['Compost', 'Trash', 'Recycle']

# Initialize webcam
cam = cv2.VideoCapture(0)

# Detection counters
compost_count = 0
trash_count = 0
recycle_count = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    img = cv2.resize(frame, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    class_name = class_labels[class_index]
    confidence = np.max(prediction) * 100

    # Counting detected items
    if class_name == 'Compost':
        compost_count += 1
        color = (0, 255, 0)  # Green
    elif class_name == 'Trash':
        trash_count += 1
        color = (0, 0, 255)  # Red
    else:  # Recycle
        recycle_count += 1
        color = (255, 0, 0)  # Blue

    # Draw a colored rectangle based on the prediction
    cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), color, 15)

    # Display Prediction Text
    text = f"{class_name} ({confidence:.2f}%)"
    cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

    # Display the counts
    cv2.putText(frame, f"Compost: {compost_count}", (10, frame.shape[0]-80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"Trash: {trash_count}", (10, frame.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(frame, f"Recycle: {recycle_count}", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    # Show the frame
    cv2.imshow("Smart Waste Classifier", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cam.release()
cv2.destroyAllWindows()
