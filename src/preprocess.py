import cv2
import os
def process(i):
        for j in os.listdir('database/'+i):
            image_path = os.path.join("database",i,j)
            image = cv2.imread(image_path)
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_image)
            v_eq = cv2.equalizeHist(v)
            hsv_eq_image = cv2.merge((h, s, v_eq))
            bgr_eq_image = cv2.cvtColor(hsv_eq_image, cv2.COLOR_HSV2BGR)
            output_path = image_path
            cv2.imwrite(output_path, bgr_eq_image)
if __name__ == "__main__":
    process('Vaibhav')