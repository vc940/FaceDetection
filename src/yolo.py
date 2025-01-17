import cv2
from ultralytics import YOLO
import os
import time

model = YOLO('model/best.pt') 
camera_index = 0
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

def start_storing(Name):

    os.makedirs(f'database/{Name}/',exist_ok=True)

    frameF= 1
    ref_time = time.time()
    while time.time() - ref_time < 10:
        boxes = ""
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break
        results = model.predict(frame, conf=0.5, show=False) 
        annotated_frame = results[0].plot()  
        if(len(results[0].boxes.xywh)>0):
            print(results[0].boxes.xywh)
            x = (results[0].boxes.xywh[0][0])
            y = (results[0].boxes.xywh[0][1])
            w = (results[0].boxes.xywh[0][2])
            h = (results[0].boxes.xywh[0][3])
            cv2.imwrite(f'database/{Name}/{frameF}.jpg',frame[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2), :])
            frameF +=1
        cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()  # Assuming you're using YOLOv8

def start_detection():
    # Create cache directory to store detected frames
    os.makedirs(f'cache', exist_ok=True)

    frameF = 1
    ref_time = time.time()
    
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # Replace 0 with video path if using a video file
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while time.time() - ref_time < 10:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Run YOLOv8 model on the frame (adjust model according to your needs)
        results = model.predict(frame, conf=0.5, show=False)  # Assuming 'model' is already initialized
        annotated_frame = results[0].plot()  # Plot bounding boxes on the frame
        
        # Process detected bounding boxes
        if len(results[0].boxes.xywh) > 0:
            # Get the first bounding box (you can loop over all if needed)
            x, y, w, h = results[0].boxes.xywh[0]
            # Ensure bounding box is within frame dimensions
            x1, y1, x2, y2 = max(0, int(x - w / 2)), max(0, int(y - h / 2)), int(x + w / 2), int(y + h / 2)
            # Crop the image based on the bounding box and save it
            cropped_face = frame[y1:y2, x1:x2]
            cv2.imwrite(f'cache/{frameF}.jpg', cropped_face)
            frameF += 1

        # Show the annotated frame
        cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_storing('Vaibhav')