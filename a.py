import pandas as pd
import ast

# Load the detections.csv file
detections_df = pd.read_csv('detections.csv')

# Initialize a list to hold the processed data
processed_data = []

# Group by frame number
grouped = detections_df.groupby('frame')

for frame, group in grouped:
    # Extract class names, confidence values, bounding boxes, and centroids
    class_names = group['class_name'].tolist()
    confidences = group['confidence'].tolist()
    bboxes = group['bbox'].tolist()
    centroids = group['centroid'].tolist()
    
    # Count the occurrences of each class
    car_count = class_names.count('car')
    truck_count = class_names.count('truck')
    motorcycle_count = class_names.count('motorcycle')
    ambulance_count = class_names.count('ambulance')
    others_count = len(class_names) - (car_count + truck_count + motorcycle_count + ambulance_count)

    # Prepare the row for the processed data
    processed_row = {
        'frame': frame,
        'class_name': str(class_names),
        'confidence': max(confidences),  # Use the maximum confidence
        'bbox': str(bboxes),
        'centroid': str(centroids),
        'velocity': group['velocity'].iloc[0],  # Use the velocity of the first detected object
        'car_count': car_count,
        'truck_count': truck_count,
        'motorcycle_count': motorcycle_count,
        'ambulance_count': ambulance_count,
        'others_count': others_count
    }
    
    processed_data.append(processed_row)

# Create a DataFrame from the processed data
processed_df = pd.DataFrame(processed_data)

# Save the processed DataFrame to help.csv
processed_df.to_csv('help.csv', index=False)

print("help.csv has been generated successfully.")