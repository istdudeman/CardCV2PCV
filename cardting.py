import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import simpledialog
from threading import Thread
import time

# Global variables to store points
points = []

# Function to order points in a consistent manner
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect

# Function to perform the four-point perspective transform
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# Mouse callback function to capture points
def select_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Points", param)

# Function to capture images and warp based on selected points
def capture_and_warp(folder_name):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot access the camera")
        return

    # Create the main directory if it doesn't exist
    main_dir = 'kartu-kartu'
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)

    # Create the specific folder for this capture
    specific_folder = os.path.join(main_dir, folder_name)
    if not os.path.exists(specific_folder):
        os.makedirs(specific_folder)

    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from camera")
        return

    # Display the frame and allow user to select points
    cv2.imshow("Select Points", frame)
    cv2.setMouseCallback("Select Points", select_points, frame)
    cv2.waitKey(0)  # Wait for key press to close the window

    if len(points) == 4:
        warped_image = four_point_transform(frame, np.array(points))
        cv2.imshow("Warped Image", warped_image)
        cv2.waitKey(0)  # Wait for key press to close the window
        # Save the warped image
        image_path = os.path.join(specific_folder, 'warped_card.png')
        cv2.imwrite(image_path, warped_image)
    else:
        print("Not enough points selected.")

    cap.release()
    cv2.destroyAllWindows()

# Function to start the image capture in a separate thread
def start_capture():
    global points
    points = []  # Reset points
    folder_name = simpledialog.askstring("Input", "Enter folder name:")
    if folder_name:
        # Start the capture in a new thread to avoid blocking the GUI
        Thread(target=capture_and_warp, args=(folder_name,)).start()

# Function to open the camera