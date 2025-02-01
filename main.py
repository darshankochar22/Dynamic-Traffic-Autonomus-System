import cv2
import numpy as np
import csv
import os
from ultralytics import YOLO

# Load a pre-trained YOLO model
model = YOLO('yolov10n.pt')

# Open the video file
video_path = '/Users/darshan/Downloads/Minor3/ambulance.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}.")
    exit()

# Get video details (width, height, frames per second)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
time_per_frame = 1 / fps  # Time between frames in seconds

# Define the codec and create VideoWriter object to save the output video
output_path = os.path.join(os.path.dirname(__file__), 'detected_output.mp4')
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Define the CSV file path in the current directory
csv_output_path = os.path.join(os.path.dirname(__file__), 'detections.csv')

# Initialize the CSV file to store detections
with open(csv_output_path, mode='w', newline='') as csv_file:
    fieldnames = ['frame', 'class_name', 'confidence', 'bbox', 'centroid', 'velocity']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Dictionary to store the previous positions of detected objects
    previous_positions = {}
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO inference on the frame
        results = model(frame)
        
        # Initialize a dictionary to hold the count of each detected class
        object_count = {}

        # Loop through detections and count objects, also calculate velocity
        for detection in results[0].boxes:
            class_id = detection.cls.cpu().numpy().astype(int)[0]
            class_name = model.names[class_id]  # Get class name using the model's names attribute
            confidence = detection.conf.cpu().numpy()[0]
            bbox = detection.xyxy.cpu().numpy().astype(int)[0].tolist()
            
            # Process only ambulances
            if class_name.lower() == "ambulance":
                # Increment the count for ambulances
                object_count[class_name] = object_count.get(class_name, 0) + 1

                # Calculate the centroid of the bounding box
                centroid = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

                # Calculate velocity if we have the previous position
                object_id = f"{class_name}-{object_count[class_name]}"  # Unique ID for the object
                if object_id in previous_positions:
                    previous_centroid, previous_frame_time = previous_positions[object_id]
                    distance = np.linalg.norm(np.array(centroid) - np.array(previous_centroid))
                    velocity = distance / time_per_frame  # Pixels per second
                else:
                    velocity = 0.0

                # Update previous position for the next frame
                previous_positions[object_id] = (centroid, cap.get(cv2.CAP_PROP_POS_MSEC))

                # Write data to CSV
                writer.writerow({
                    'frame': frame_number,
                    'class_name': class_name,
                    'confidence': f"{confidence:.2f}",
                    'bbox': bbox,
                    'centroid': centroid,
                    'velocity': f"{velocity:.2f}"
                })

                # Draw the bounding box and label on the frame
                label = f"{class_name} {confidence:.2f}"
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)  # Red box for ambulance
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                velocity_label = f"Velocity: {velocity:.2f} px/s"
                cv2.putText(frame, velocity_label, (bbox[0], bbox[3] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Write the annotated frame to the output video
        out.write(frame)

        # Optionally display the frame with detections
        cv2.imshow('YOLO Detection - Ambulance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
