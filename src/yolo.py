import cv2
from ultralytics import YOLO

model = YOLO('model/best.pt') 
camera_index = 0
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

def start_storing():
    frameF= 1

    while True:
        boxes = ""

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break
        results = model.predict(frame, conf=0.5, show=False) 
        annotated_frame = results[0].plot()  
        if(len(results[0].boxes.xywh)>0):
            boxes += str(frameF)+" "+ " ".join(map(str, results[0].boxes.xywh[0].tolist())) + "\n"
            print(results[0].boxes.xywh)
            x = (results[0].boxes.xywh[0][0])
            y = (results[0].boxes.xywh[0][1])
            w = (results[0].boxes.xywh[0][2])
            h = (results[0].boxes.xywh[0][3])
            cv2.imwrite(f'database/Vaibhav/{frameF}.jpg',frame[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2), :])
            frameF +=1
            with open('labels.txt','a') as file:
                file.write(boxes)
        cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    start_storing()