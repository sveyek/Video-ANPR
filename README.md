# Automatic Number Plate Recognition in Video Streams 

This project employs YOLOv8, trained on the COCO dataset, for general object detection, and a fine-tuned YOLO model specifically designed for license plate detection. The aim is to track vehicles and identify license plates within video streams. Subsequently, EasyOCR is utilized to extract text from the detected license plates.
Output from the application is show below.
<center>
    <img src="https://github.com/SevinduEk/Video-ANPR/assets/81402530/75fee3cb-8485-45c5-b3e7-3d17cce4f6dd" alt="output" width="800">
</center>

## Procedure Overview:

1. **Vehicle and License Plate Detection**: 
   - Utilized the COCO-trained YOLO model to identify vehicles.
   - Deployed a fine-tuned YOLO model to specifically detect license plates.

2. **Association of Plates to Vehicles**:
   - Mapped each detected license plate to its corresponding vehicle by comparing the positions of their respective bounding boxes.

3. **Text Extraction**:
   - Extracted text from the detected license plate areas for each frame using EasyOCR.

4. **Error Handling**:
   - Addressed potential misreadings from the OCR model.
   - Filtered out incorrect readings by comparing them against the standard license number format. (Note: The demo utilizes UK number plates.)

5. **Enhancements**:
   - Interpolated detections to provide a smoother and visually improved output.

6. **Final Output**:
   - For each vehicle, the license plate reading with the highest confidence score was selected.
   - Displayed the interpreted license plates within the video output.

Resources: 
- Dataset: [License Plate Recognition Dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4)
