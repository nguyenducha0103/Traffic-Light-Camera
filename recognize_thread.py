import threading
import ctypes
import cv2
import time
import numpy as np
import glob
import os

# from onnx_face.FaceDetector import FaceDetector
# from onnx_face.FaceExtractor import FaceExtractor


from face_models.face_align import norm_crop
from tools.search import VectorWarehouse, compute_sim

class ExtractionThread(threading.Thread):
    def __init__(self, queue_person):
        threading.Thread.__init__(self )
        # self.id = self.camera.get_streamID()

        self.stream_capture = ''
        self.name = 'ExtractionThread'
        self.stop_flag = False
        # self.extractor = FaceExtractor()
        # self.extractor = TritonExtractor()
        # _, self.jpeg_stopimage = cv2.imencode('.jpg',self.stopstream_image) 
        self.queue_person = queue_person
        

        self.emb_lst = []
        self.ID_dict = {}
        self.name_dict = {}
        # self.extractor = FaceExtractor()
        
        

        # self.vector_wh = VectorWarehouse(vector_list=np.array(self.emb_lst), dimension=512)
        # self.camera_config = {}
        # print(type(self.queue_frame['source']))

    def run(self):
        while not self.stop_flag:
            
            time.sleep(0.001)
            
        
        # time.sleep(0.01)

    def get_id(self):
        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure') 

    def stop(self):
        self.stop_flag = True
        # self.ffmpegrestream.stop()
    # def get_camera_id(self):