import time
import cv2
import time
import threading
import ctypes
# from ffmpeg_restream import FfmpegRestream

class CameraReading(threading.Thread):
    def __init__(self, queue_camera):
        threading.Thread.__init__(self)
        self.queue_camera = queue_camera
        # self.id = self.camera.get_streamID()
        self.stream_capture = ''
        self.name = 'CameraReading'
        self.stop_flag = False

        self.stopstream_image = cv2.imread("endstream.jpg")
        # _, self.jpeg_stopimage = cv2.imencode('.jpg',self.stopstream_image) 
        
        # self.camera_config = {}
        # print(type(self.queue_camera['source']))
        self.stream_capture = cv2.VideoCapture(self.queue_camera['source'])
        
        self.fps = 30
        self.restream_flag = False
        
        self.stream_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.fps = self.stream_capture.get(cv2.CAP_PROP_FPS)
        self.sleep_time = 0.02
        print('self.fps:',self.fps)

    def run(self):
        while not self.stop_flag:
            
            if self.stream_capture.isOpened():
                ret, frame = self.stream_capture.read()
                # self.stream_capture.grab()
                # ret, frame = self.stream_capture.retrieve()
                if not ret:
                    self.stream_capture = cv2.VideoCapture(self.queue_camera['source'])
                    time.sleep(self.sleep_time)
                    frame = self.stopstream_image
                
            else:
                frame = self.stopstream_image
            # print('frame:',frame)
            self.queue_camera['frame'] = frame
            # cv2.imshow('img',self.queue_camera)
            # cv2.waitKey(1)
            time.sleep(self.sleep_time)

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
    #     return self.id


    