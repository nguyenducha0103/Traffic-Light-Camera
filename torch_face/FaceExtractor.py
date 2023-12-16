import cv2
import numpy as np
from onnx_face.onnx_runtime import ONNXBase

from numpy.linalg import norm

def l2_norm(input, axis = 1):
    normed = norm(input)
    output = input/normed
    return output

class FaceExtractor(ONNXBase):
    def __init__(self, weight_path, input_shape=(112, 112)):
        super().__init__(weight_path)

        self.input_shape = input_shape

    def preprocess(self, image):
        input_width, input_height = self.input_shape

        img = cv2.resize(image, (input_width, input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_flip = cv2.flip(img, 1)
        # cv2.imwrite('img_flip.jpg', img_flip)
        imgs = [img, img_flip]
            
        imgs_processed = []
        
        for im in imgs:
            im = np.transpose(im, (2, 0, 1))
            im = (im / 255. - 0.5) / 0.5
            im = np.float32(im)
            
            imgs_processed.append(im)
            
        img_batch = np.stack((imgs_processed[0], imgs_processed[1]))
        
        return img_batch
    

    def get_embedding(self, input_image):
        
        input_image = self.preprocess(input_image)
        result = self.infer(input_image)
        # embeding_concat = np.concatenate((result[0].reshape(-1),result[1].reshape(-1)))
        embeding_add = result[0].reshape(-1) + result[1].reshape(-1)
        
        l2n = l2_norm(embeding_add)
        
        return l2n
    
    def compute_sim(self, emb1, emb2):
        sim = np.dot(emb1, emb2)/(norm(emb1)*norm(emb2))
        return sim
    
# if __name__ == "__main__":
#     model = FaceExtractor(weight_path='app/client/face_recognition/onnx_face/weights/FaceExt.onnx')
#     import time

#     image = cv2.imread('duc.jpg')

#     for i in range(100):
        
#         t1 = time.time()

#         embs = model.get_embedding(image)
#         t2 = time.time()
#         print('Time total', t2 - t1)