import cv2
import numpy as np

# Function to calculate distance
def calculate_distance(object_width, focal_length, apparent_width):
    return (object_width * focal_length) / apparent_width

# Known size of the object in the real world (in this example, 10 cm)
object_width = 10.0  # in centimeters

# Open a video capture object
cam = cv2.VideoCapture(0)

# Read two initial frames
ret, frame1 = cam.read()
ret, frame2 = cam.read()

# Camera calibration parameters (you need to perform camera calibration first)
# Example parameters (replace with your calibrated values)
focal_length = 1000.0  # Example focal length in pixels
# ...

while cam.isOpened():
    # Read a new frame
    ret, frame1 = cam.read()
    ret, frame2 = cam.read()

    # Calculate the absolute difference between the two frames
    diff = cv2.absdiff(frame1, frame2)

    # Convert the difference to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply a threshold to create a binary image
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Dilate the thresholded image to fill gaps in contours
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours and draw rectangles around moving objects
    for c in contours:
        if cv2.contourArea(c) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Calculate distance using triangulation
        if len(contours) >= 3:
            # Assuming the object is a rectangular shape, use the width of the bounding box
            distance = calculate_distance(object_width, focal_length, w)
            # Display distance information on the frame
            cv2.putText(frame1, f"Distance: {distance:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the result
    cv2.imshow('Motion Detection with Distance', frame1)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(10) == ord('q'):
        break

# Release the capture object
cam.release()
cv2.destroyAllWindows()
