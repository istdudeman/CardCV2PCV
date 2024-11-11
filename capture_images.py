import cv2
import numpy as np
import os

# Define dataset path and number of images to capture per card
dataset_path = "dataset/train"  # Directory where your dataset will be saved
num_images = 100  # Number of images to capture per card
card_label = "2"  # Label for the card you are capturing (e.g., "2", "3", "K")

# Path for saving images
save_path = os.path.join(dataset_path, card_label)

# Create directory if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Access the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize a counter
img_count = 0

print(f"Capturing {num_images} images for card: {card_label}")
print("Press 's' to start capturing images, or 'q' to quit.")

# Define lower and upper bounds for the color to detect (adjust this as needed)
lower = np.array([40, 40, 40])  # Example for green
upper = np.array([80, 255, 255])  # Example for green

# Kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)  # Adjust size as needed

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for detecting the desired color
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_not(mask)  # Invert the mask to get the foreground

    # Perform erosion and dilation to reduce noise
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Extract only the foreground from the frame
    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the frame with the foreground
    cv2.imshow("Capture Image", frame)
    cv2.imshow("Foreground", foreground)

    # Wait for 's' key to start capturing or 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        print("Starting capture...")
        while img_count < num_images:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break
            
            # Convert frame to HSV and create mask again
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_not(mask)
            mask = cv2.erode(mask, kernel, iterations=2)
            mask = cv2.dilate(mask, kernel, iterations=2)

            # Check if the foreground has significant area
            if cv2.countNonZero(mask) > 1000:  # Adjust threshold as needed
                # Save image to the corresponding folder
                img_name = f"{card_label}_{img_count + 1}.jpg"
                img_path = os.path.join(save_path, img_name)
                cv2.imwrite(img_path, frame)
                img_count += 1
                print(f"Captured {img_name}")

                # Display the capture count
                cv2.putText(frame, f"Captured {img_count}/{num_images}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Capture Image", frame)

            # Brief pause to simulate the next capture
            cv2.waitKey(100)

        print("Capture complete!")
        break

    elif key == ord('q'):
        print("Capture canceled.")
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

