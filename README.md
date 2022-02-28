# Licence-Plate-Recognition-Real-Time
This project aims to detect license plates in real-time images and videos using yolov5 and multiprocessing. 

"THINK989/Automatic_License_Plate_Detector_for_Adverse_Environments" was used in the project to detect plates. Thank you.

Here you can visit from: https://github.com/THINK989/Automatic_License_Plate_Detector_for_Adverse_Environments 

Requirements

-YoloV5
-Tesseract OCR
-OpenCV
-Cuda and Cudnn

Using

    1- conda create --name LicencePlate python=3.7 anaconda
    2- git clone https://github.com/THINK989/Automatic_License_Plate_Detector_for_Adverse_Environments 
    3- cd yolov5
    4- pip install -r requirements.txt
    5- pip install pytesseract 
    6- pip install imutils

-Download this repo after you have done all the necessary installations. 
-Then replace the files in this repo with the files from the original project.

Files to be replaced and their locations: 
-detect.py ----> in yolov5
-datasets.py ----> in yolov5/utils

Test Environment
- Ubuntu 20.04
- Python 3.7.11
- OpenCv 4.5.5
- Numpy 1.17.3
- Pytesseract 0.3.9
- CUDA 11.2
- GTX 1650

A helpful resource for installing cuda and cudnn: 
    
- https://medium.com/analytics-vidhya/install-cuda-11-2-cudnn-8-1-0-and-python-3-9-on-rtx3090-for-deep-learning-fcf96c95f7a1
