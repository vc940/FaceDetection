from deepface import DeepFace
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def find_person(image_path):
    result = DeepFace.find(
        img_path=image_path,
        db_path="database",
        model_name="VGG-Face",
        detector_backend="mtcnn"  
    )
    return (result[0]['identity'].apply(lambda x: x.split('/')[-2]).value_counts())
    
if __name__ =='__main__':
    person =  find_person('image.png')