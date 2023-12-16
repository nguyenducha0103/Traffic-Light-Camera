import tritonclient.grpc as grpcclient
import numpy as np
import cv2
from numpy.linalg import norm

def l2_norm(input, axis = 1):
    normed = norm(input)
    output = input/normed
    return output

def d_process(im):
    im = np.transpose(im, (2, 0, 1))
    im = (im / 255. - 0.5) / 0.5
    return im

def preprocess(image, input_width, input_height):
    image = np.float32(image)
    
    # img = cv2.resize(image, (input_width, input_height))
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    img_flip = cv2.flip(img, 1)
    
    im1 = d_process(img)        
    
    return im1

def preprocess_batch(batch_image, input_width, input_height):
    batch_image = np.array([preprocess(image, input_width, input_height) for image in batch_image]) 
    print(batch_image.shape)
    return batch_image

class TritonExtractor():
    def __init__(self,
            model = 'face_extraction_b10',
            input_width = 112,
            input_height = 112,
            mode = "FP32",
            url = '10.70.39.40:8011',
            verbose = False,
            ssl = None,
            root_certificates = None,
            private_key = None,
            certificate_chain = None,
            client_timeout = None,
            batch_size = 10):
            
        self.model = model
        
        self.inputs = []
        self.outputs = []
        self.input_width = input_width
        self.input_height = input_height
        self.mode = mode
        self.batch_size = batch_size
        
        self.inputs.append(grpcclient.InferInput('input', [self.batch_size, 3, self.input_height, self.input_width], self.mode))
        self.outputs.append(grpcclient.InferRequestedOutput('ouput'))
        
        self.client_timeout = client_timeout

        # Create server context
        self.triton_client = grpcclient.InferenceServerClient(
            url=url,
            verbose=verbose,
            ssl=ssl,
            root_certificates=root_certificates,
            private_key=private_key,
            certificate_chain=certificate_chain)
        
    def get_embedding_batch(self, input_batch_image):
        
        input_image_buffers = preprocess_batch(input_batch_image, self.input_width, self.input_height)
        print(input_image_buffers.shape)
        self.inputs[0].set_data_from_numpy(input_image_buffers)

        results = self.triton_client.infer(model_name=self.model,
                                    inputs=self.inputs,
                                    outputs=self.outputs,
                                    client_timeout=self.client_timeout)
        
        result = results.as_numpy('ouput')
        print(result.shape)
        # embeding_concat = np.concatenate((result[0].reshape(-1),result[1].reshape(-1)))
        
        
        return result

    
    def get_embedding(self, input_image):
        
        input_image_buffers = preprocess(input_image, self.input_width, self.input_height)
        
        self.inputs[0].set_data_from_numpy(input_image_buffers)

        results = self.triton_client.infer(model_name=self.model,
                                    inputs=self.inputs,
                                    outputs=self.outputs,
                                    client_timeout=self.client_timeout)
        
        result = results.as_numpy('ouput')
        # print(result.shape)
        # embeding_concat = np.concatenate((result[0].reshape(-1),result[1].reshape(-1)))
        embeding_add = result[0].reshape(-1) + result[1].reshape(-1)
        
        l2n = l2_norm(embeding_add)
        
        return l2n

def compute_sim(emb1, emb2):
    sim = np.dot(emb1, emb2)/(norm(emb1)*norm(emb2))
    return sim

if __name__ == '__main__':
    import time
    extractor = TritonExtractor()
    
    img1 = np.zeros((112,112,3))
    batch = [img1 for i in range(10)]
    for i in range(10):
        t1 = time.time()
        r1 = extractor.get_embedding_batch(batch)
        t2 = time.time()
        print('time', t2 - t1)