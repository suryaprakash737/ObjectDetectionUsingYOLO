# Object Detection Using YOLO

This project demonstrates real-time and image-based object detection using the YOLO (You Only Look Once) algorithm with OpenCV and pre-trained weights. The repository contains scripts for detecting objects in images, videos, and via webcam feed.

---

## Project Files

1. theone-image.py
   - Detects objects in static images.
   - Utilizes the YOLOv3 model and OpenCV for processing.
   - Example images included in the repository.

2. theone-yolov3-tiny.py
   - Processes video files for object detection.
   - Capable of handling real-time object detection from pre-recorded videos.
   - Outputs bounding boxes and labels for detected objects.

3. webcam.py 
   - Detects objects in real-time using a webcam feed.
   - Optimized for real-time performance using the YOLOv3 configuration and weights.

---

## Installation

### Prerequisites
- Python 3.x
- OpenCV (`cv2`)
- NumPy

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/suryaprakash737/ObjectDetectionUsingYOLO.git
   cd ObjectDetectionUsingYOLO
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the following files are present:
   - `yolov3.cfg`
   - `yolov3.weights`
   - `coco.names`

---

## How to Run

### 1. Image Detection
   ```bash
   python theone-image.py
   ```
   - Modify the script to select the desired image (`img = cv2.imread("<image_name>")`).

### 2. Video Detection
   ```bash
   python theone-yolov3-tiny.py
   ```
   - Update the `video_capture` variable to the desired video file path.

### 3. Webcam Detection
   ```bash
   python webcam.py
   ```
   - Ensures your webcam is connected.

---

## Features
- Real-time object detection with bounding boxes and labels.
- Image and video file support.
- Simple and modular Python scripts.

---

## Acknowledgments
- YOLOv3 architecture by Joseph Redmon and Ali Farhadi.
- Pre-trained weights and configuration files available at [YOLO Official Site](https://pjreddie.com/darknet/yolo/).

---

## License
This project is licensed under the MIT License. Feel free to use and modify it for educational purposes.

