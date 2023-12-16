
from onnx_face.FaceExtractor import FaceExtractor
from onnx_face.FaceDetector import FaceDetector
from face_models.face_align import norm_crop

from face_models.tritonExtractor import TritonExtractor
from face_models.tritonDetector import TritonDetector

import glob
import cv2
import numpy as np
d = FaceDetector()
e = FaceExtractor()

import os

def draw(img, bbox, lmk):
    for b, l in zip(bbox, lmk):
        box = [int(x) for x in b[:4]]
        
        cv2.rectangle(img,(box[0], box[1]), (box[2], box[3]), (255,255,0),1 )
        cv2.putText(img, str(b[4]),(box[0], box[1]-12),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (110,0,110), 1,cv2.LINE_AA )     
        for p in l:
            p = [int(x) for x in p]
            
            cv2.circle(img, (p[0], p[1]), 1, (0, 0, 255), 4)
            cv2.circle(img, (p[0], p[1]), 1, (0, 0, 255), 4)
            cv2.circle(img, (p[0], p[1]), 1, (0, 0, 255), 4)
            cv2.circle(img, (p[0], p[1]), 1, (0, 0, 255), 4)
            cv2.circle(img, (p[0], p[1]), 1, (0, 0, 255), 4)

def extract(storage_path, emb_path):
    for person_path in glob.glob(storage_path):
        name_person = os.path.basename(person_path)
        # name_person = name_img.split('.jpg')[0]
        print(name_person)
        os.makedirs(os.path.join(emb_path, name_person), exist_ok=True)
        
        for img_path in glob.glob(os.path.join(person_path, '*')):
            print(img_path)
            img = cv2.imread(img_path)
            name_img = os.path.basename(img_path).split('.')[0]

            bbox, lmk = d.detect(img)
            
            for b, l in zip(bbox, lmk):
                face_img = norm_crop(img, l)
                emb = e.get_embedding(face_img)

                path_save = os.path.join(emb_path, name_person, name_img + '.npy')
                
                np.save( path_save, emb)
    
extract('./person_list/*', './emb_storage/emb_folder_onnx/')