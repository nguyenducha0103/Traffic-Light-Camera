import threading
import ctypes
import cv2
import time
import numpy as np
from numpy import asarray

from face_models.tritonDetector import TritonDetector
from vehicle_models.detector import TritonVehicle

from tracking.bytetrack.byte_tracker import BYTETracker
from tracking.sort import Sort
from object_manager import VehicleManager



detector = TritonVehicle()

tracker = BYTETracker()
# tracker = Sort(max_age=2, min_hits=4, max_step_recog=30)
vehicle_manager = VehicleManager()

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def draw(img, bbox, lmk, id):
    for b, l, i in zip(bbox, lmk, id):
        box = [int(x) for x in b[:4]]
        # conf = str(b[4])
        cv2.rectangle(img,(box[0], box[1]), (box[2], box[3]), (255,255,0),2 )
        # img = draw_box(img, box)
        # cv2.putText(img, conf,(box[0], box[1]-12),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (110,0,110), 1,cv2.LINE_AA )     
        cv2.putText(img, str(i),(box[0], box[1]-12),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,110), 1,cv2.LINE_AA )     
    return img


class ProcessThread(threading.Thread):
    def __init__(self, queue_frame, queue_camera, queue_person, queue_restream):
        threading.Thread.__init__(self)
        self.queue_frame = queue_frame
        self.queue_camera = queue_camera
        self.queue_person = queue_person
        self.queue_restream = queue_restream
        # self.id = self.camera.get_streamID()

        self.name = 'ProcessThread'
        self.stop_flag = False

        self.stopstream_image = cv2.imread("endstream.jpg")
        # _, self.jpeg_stopimage = cv2.imencode('.jpg',self.stopstream_image) 
        
        # self.camera_config = {}
        # print(type(self.queue_frame['source']))

    def run(self):
        while not self.stop_flag:
            if len(self.queue_camera):
                frame = self.queue_camera.popleft()
                # frame = asarray(frame)
                t1 = time.time()
                
                frame = image_resize(frame, width=1280, height=720)
                detections = detector.detect(frame)

                if len(detections):
                    # clss = np.ones((len(bbox))).astype(np.int32)
                    bboxes, scores, classes = detections[:,:4], detections[:,4], detections[:,5]
                    
                    track_result = tracker.update(bboxes,scores, classes)

                    # track_sort = []
                    # for track in track_result:
                    #     x1,y1,x2,y2 = [int(x) for x in track[0]]
                    #     id = track[1]
                    #     lmk = track[2]
                    #     track_sort.append(np.array([x1,y1,x2,y2,id, lmk]))

                    frame = vehicle_manager.update_tracking(track_result, frame, self.queue_person)
                
                t2 = time.time()
                # print(t2 - t1)
                frame = vehicle_manager.putText_utf8(frame, f'FPS: {int(1/(t2 - t1))}',(10, 50), ((251,104,223)))
                # frame = cv2.putText(frame, f'FPS: {int(1/(t2 - t1))}',(10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1,cv2.LINE_AA )   
                
                self.queue_frame.append(frame)
                self.queue_restream.append(frame)
            else:
                time.sleep(0.03)

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