import cv2
import os
import tkinter as tk
from tkinter import messagebox
from datetime import datetime

class CardCaptureApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Foto kartu")

        # Create a button to capture an image
        self.capture_button = tk.Button(master, text="Capture Image", command=self.capture_image)
        self.capture_button.pack(pady=20)

        # Initialize video capture
        self.video_capture = cv2.VideoCapture(0)

        # Create a directory for saving images
        self.main_dir = "kartu-kartu"
        if not os.path.exists(self.main_dir):
            os.makedirs(self.main_dir)

    def capture_image(self):
        success, frame = self.video_capture.read()
        if success:
            # Create a timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"card_{timestamp}.jpg"

            # Save the captured image
            cv2.imwrite(os.path.join(self.main_dir, filename), frame)
            messagebox.showinfo("Success", f"Image saved as {filename} in {self.main_dir}")
        else:
            messagebox.showerror("Error", "Failed to capture image.")

    def on_closing(self):
        self.video_capture.release()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CardCaptureApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()