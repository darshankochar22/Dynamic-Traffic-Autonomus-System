# Dynamic-Traffic-Autonomus-System

## Overview
This project leverages **YOLO (You Only Look Once)**, a powerful real-time object detection algorithm, to analyze traffic patterns from video input. Our algorithm processes **.mp4** files as input, detects vehicles and other objects using YOLO, and generates structured data in a **CSV file**. With this data, we aim to analyze and optimize **dynamic traffic signal timings** to improve urban traffic flow.

## Features
- **Real-time Object Detection**: Utilizes YOLO to detect vehicles, pedestrians, and other road objects.
- **CSV-based Data Analysis**: Extracted detections are stored in a CSV file for further analysis.
- **Dynamic Signal Timing Optimization**: Using the generated CSV data, our algorithm assesses traffic congestion and suggests optimized signal timings.
- **FastAPI Deployment**: The project is designed to be easily deployable using FastAPI for seamless API-based integration.

## Installation
```bash
# Clone the repository
git clone https://github.com/your-repo-link.git
cd your-repo-name

# Install dependencies
pip install -r requirements.txt
```

## Usage
1. **Run the main script:**
   ```bash
   python main.py --input video.mp4
   ```
2. **Check the output CSV file:**
   - The detected objects and their coordinates will be saved in `synthetic_detections_file_2.csv`.
   - This data will be used for further analysis on optimizing traffic signal timing.

## File Structure
```
📂 Project Directory
│── main.py                # Main execution script
│── a.py                   # Supporting script
│── bombay.mp4             # Sample input video
│── detected_output.mp4    # Processed output video
│── synthetic_detections_file_2.csv # YOLO detected output
│── README.md              # This file
```

## About YOLO
**YOLO (You Only Look Once)** is a state-of-the-art real-time object detection algorithm. Unlike traditional approaches that use sliding windows or region proposals, YOLO processes an entire image in one forward pass of a deep neural network, making it extremely fast and efficient.

### Why YOLO for Traffic Analysis?
- High-speed object detection in real-time.
- Accurate recognition of vehicles, pedestrians, and other objects.
- Suitable for traffic monitoring applications where quick decisions are needed.

## Community & Acknowledgment
This project was made possible with the contributions of the open-source community and advancements in computer vision research. We extend our gratitude to:
- **The YOLO Community** for open-source contributions.
- **FastAPI Developers** for enabling seamless deployment.
- **Researchers & Developers** working towards smarter traffic management solutions.

## Future Scope
- Implement **real-time adaptive traffic light control** based on congestion analysis.
- Integrate **machine learning models** to predict traffic patterns.
- Expand detection to include emergency vehicles for priority-based signaling.

## Contributors
- [Your Name]
- [Your Team Members]

## License
This project is licensed under the **MIT License**.

---
✨ *Optimizing traffic, one signal at a time!* 🚦

