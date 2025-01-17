from deepface import DeepFace
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def find_person():
    result = DeepFace.find(
        img_path="image.png",
        db_path="FaceDetection/database",
        model_name="VGG-Face",
        detector_backend="mtcnn"  
    )
    print(result[0]['identity'].apply(lambda x: x.split('/')[-2]).value_counts())
    
if __name__ =='__main__':
    find_person()