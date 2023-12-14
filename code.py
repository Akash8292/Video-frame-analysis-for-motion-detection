
import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture('video.mp4')  

# Create background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Thresholding to get binary image
    _, thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around moving objects
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Adjust this threshold as needed
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the original frame and the motion detection result
    cv2.imshow('Original', frame)
    cv2.imshow('Motion Detection', thresh)

    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()


