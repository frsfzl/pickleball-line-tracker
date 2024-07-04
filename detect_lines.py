import cv2
import numpy as np

def process_frame(frame):
    # Get the height and width of the frame
    height, width = frame.shape[:2]
    
    # Define the region of interest (bottom 55% of the frame)
    roi_start = int(height * 0.45)
    roi = frame[roi_start:height, :]
    
    # Convert the ROI to HSV color space for better color filtering
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define the range for white color in HSV space
    lower_white = np.array([0, 0, 200])  # Lower bound for white
    upper_white = np.array([180, 30, 255])  # Upper bound for white
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Apply the mask to the frame to get only white regions
    white_regions = cv2.bitwise_and(roi, roi, mask=mask)
    
    # Convert the white regions to grayscale
    gray = cv2.cvtColor(white_regions, cv2.COLOR_BGR2GRAY)
    
    # Apply a Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    
    # Use Canny edge detection to find edges
    edges = cv2.Canny(opening, 50, 150)
    
    # Use the Hough Line Transform to detect lines in the image
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=30)
    
    # Draw the detected lines on the original frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Adjust the y-coordinates back to the original frame
            y1 += roi_start
            y2 += roi_start
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return frame

def process_video(input_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    
    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get the video writer initialized to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        processed_frame = process_frame(frame)
        
        # Write the frame to the output video
        out.write(processed_frame)
        
        # Resize the frame to fit within the window
        frame_height, frame_width = processed_frame.shape[:2]
        window_size = (frame_width, frame_height)
        
        # Display the frame
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Frame', window_size[0], window_size[1])
        cv2.imshow('Frame', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture and writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Path to the input and output video files
input_video_path = '07011.mp4'
output_video_path = 'output_video.avi'

# Process the video
process_video(input_video_path, output_video_path)
