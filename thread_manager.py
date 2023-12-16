from recognize_thread import ExtractionThread
from process_thread import ProcessThread
from restreaming.camera_read import CameraReading
from restreaming.ffmpeg_restream import FfmpegRestream

class ThreadManager():
    def __init__(self, source, queue_camera, queue_frame, queue_person, queue_restream):
        self.cameraread = CameraReading(source, queue_camera)
        self.process = ProcessThread(queue_camera=queue_camera, queue_frame=queue_frame, queue_person = queue_person, queue_restream= queue_restream)
        self.extract = ExtractionThread(queue_person)
        self.restream = FfmpegRestream(queue_frame)

        self.extract.daemon = True
        self.process.daemon = True
        self.cameraread.daemon = True
        self.restream.daemon = True

    def start(self):
        self.cameraread.start()
        self.process.start()
        # self.extract.start()
        # self.restream.start()