import yolo
import similarity
import preprocess
import cv2 as cv
import os
import shutil
import random
import time


class FaceDetection:
    def __init__(self,Name):
        self.name = Name
        self.path = 'dataset/'+Name
    def add_user(self):
        yolo.start_storing(self.name)
        print('user_added')
    def refine_database(self):
        preprocess.process(self.name)    
        print('dataset_refined')
    def authenticate(self):
        os.makedirs('cache',exist_ok=True)
        yolo.start_detection()
        print( os.listdir('cache'))
        temp = 0
        for images in random.sample(os.listdir('cache'),10):
            imgpath = os.path.join('cache',images)
            self.result = similarity.find_person(imgpath)
            temp += self.result
        print(temp)
        print("user_authenticated")
    def spoof_check(self):
        pass


def main():
    a = str(input())
    detector = FaceDetection(a)
    detector.add_user()
    detector.refine_database()
    time.sleep(2)
    detector.authenticate()
    shutil.rmtree('cache')
   
if __name__ == '__main__':
    main()   
