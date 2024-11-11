import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import os

class CardWarpApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Card Warp Application")
        
        # Create a label to display the video feed
        self.video_label = tk.Label(master)
        self.video_label.pack()

        # Create a button to capture and warp the card
        self.capture_button = tk.Button(master, text="Capture and Warp Card", command=self.capture_and_warp)
        self.capture_button.pack()

        # Initialize video capture
        self.video_capture = cv2.VideoCapture(0)

        # Start the video loop
        self.update_video()

    def update_video(self):
        success, frame = self.video_capture.read()
        if success:
            # Convert frame to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.master.after(10, self.update_video)  # Update video every 10 ms

    def capture_and_warp(self):
        success, frame = self.video_capture.read()
        if not success:
            print("Failed to capture frame.")
            return

        # Convert to HSV for green screen masking
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the range for the green color
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])

        # Create a mask for the green color
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Invert the mask to get the card area
        mask_inv = cv2.bitwise_not(mask)
        result = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # Convert to grayscale and find contours
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Assume the largest contour is the card
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            if len(approx) == 4:  # We found a quadrilateral
                # Get the points for warping
                points = approx.reshape(4, 2)
                width = 300  # Desired width of the warped card
                height = 450  # Desired height of the warped card

                # Define the destination points for the warp
                dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')

                # Get the perspective transform matrix
                M = cv2.getPerspectiveTransform(points.astype('float32'), dst)
                warped = cv2.warpPerspective(frame, M, (width, height))

                # Show the warped image
                cv2.imshow("Warped Card", warped)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Could not find a quadrilateral contour.")
        else:
            print("No contours found.")

    def on_closing(self):
        self.video_capture.release()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CardWarpApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()