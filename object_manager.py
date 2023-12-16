import time
import cv2
from face_models.face_align import norm_crop
from PIL import ImageFont, ImageDraw, Image
from time import strftime, localtime
# import taichi as ti
import numpy as np

# ti.init(arch=ti.cpu)



class Vehicle():
    def __init__(self, id):
        self.id = -1
        
        self.track_id = id
        self.vehicle = None
        self.bbox = []

        self.first_time = 0
        self.last_time = 0
        self.delete_moment = 0
        
        self.lp_image = None
        self.identied = False



class VehicleManager():
    def __init__(self):
        self.list_vehicle = []
        self.dict_vehicle = {}

        self.current_time = time.time()
        self.font = ImageFont.truetype("font/RobotoSlab-Regular.ttf", 20)

        self.temporary_delete = {}
        # self.emb_lst = []
        # self.name_lst = []
        # self.extractor = FaceExtractor()

        # lst_emb = glob.glob('/face_service/emb_onnx/*')
        # for emb_np in lst_emb:
        #     name = os.path.basename(emb_np).split('.')[0]
        #     self.emb_lst.append(np.load(emb_np))
        #     self.name_lst.append(name)
        

    def add_vehicle(self, vehicle):
        self.list_vehicle.append(vehicle)
        self.dict_vehicle.update({str(vehicle.track_id): vehicle})

    # @measure_time
    def update_tracking(self, tracking_results, frame, queue_vehicle):

        self.current_time = time.time()
        # self.current_datetime = datetime.datetime.now()
        for track in tracking_results:
            # track_bbox = track[0]
            track_bbox = np.array(track[:4]).astype(np.int32)
            x1,y1,x2,y2 = track_bbox
            track_id = track[4]
            vehicle_id = str(track_id)
            # print(track_lmk)

            # update vehicle info if vehicle in queue
            if vehicle_id in self.dict_vehicle:
                vehicle = self.dict_vehicle[vehicle_id]
                vehicle.bbox = track_bbox

                vehicle.last_time = time.time()
                vehicle.vehicle_image = frame[y1:y2,x1:x2]
            # create new vehicle info if tracking not in queue
            else:
                new_vehicle_id = vehicle_id
                vehicle = Vehicle(new_vehicle_id)
                vehicle.bbox = track_bbox

                vehicle.first_time = time.time()
                vehicle.last_time = time.time()
                
                try:
                    vehicle.vehicle_image = frame[y1:y2,x1:x2]
                except:
                    print('Cant label for vehicle')
                ### add vehicle 
                self.add_vehicle(vehicle)
            
        frame = self.recognition(frame, queue_vehicle)
        return frame
    
    def update_recognition_info(self, recog_results):
        for res in recog_results:
            pass    

    def putText_utf8(self, img, text, pos, color):
        x, y = pos
        # color = np.array(color)[:-1]
        # print(color)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img)

        # Draw non-ascii text onto image
        draw = ImageDraw.Draw(pil_image)
        draw.text((x,y-25), text, font=self.font, fill=color)

        # Convert back to Numpy array and switch back from RGB to BGR
        image = np.asarray(pil_image)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image

    
    def cvt_color_putText_BGR2RGB(self, color):
        b,g,r = color
        return (r, g, b)
    
    def retracking(self, vehicle_new, vehicle_old):
        vehicle_new.vehicle_image = vehicle_old.vehicle_image
        vehicle_new.first_time = vehicle_old.first_time

    # @ti.kernel
    def wrapper(self, queue_vehicle, frame):
        for vehicle in self.list_vehicle:
            if not vehicle.identied:
                queue_vehicle.append(vehicle)

            if vehicle.id in self.temporary_delete.keys():
                print('retracking vehicle')
                old_vehicle = self.temporary_delete[vehicle.id]
                self.retracking(vehicle, old_vehicle)
                self.temporary_delete.pop(vehicle.id)
                vehicle.delete_moment = 0
            
            if time.time() - vehicle.last_time > 2:
                if vehicle.identied:
                    vehicle.delete_moment = time.time()
                    self.temporary_delete.update({vehicle.id:vehicle})
                self.dict_vehicle.pop(str(vehicle.track_id))
                self.list_vehicle.remove(vehicle)


            if time.time() - vehicle.last_time < 0.1:
                # frame = cv2.putText(frame, f'{vehicle.id}', (vehicle.bbox[0]-13,vehicle.bbox[1]),cv2.FONT_HERSHEY_SIMPLEX ,0.5, (255, 0, 150),1)
                  # cv2.imwrite('check_face.png', frame[vehicle.bbox[1]: vehicle.bbox[3], vehicle.bbox[0]: vehicle.bbox[2]])
                if not vehicle.identied:
                    cv2.rectangle(frame,(vehicle.bbox[0], vehicle.bbox[1]), (vehicle.bbox[2], vehicle.bbox[3]), (0, 20, 153),2 )
                    frame = self.putText_utf8(frame, f'Stranger', (vehicle.bbox[0],vehicle.bbox[1]), (0, 15, 153))
                
                    # frame = self.putText_utf8(frame, f'{vehicle.name}', (vehicle.bbox[0],vehicle.bbox[1]),(197, 204, 100))
                else:
                    # print('dra')
                    cv2.rectangle(frame,(vehicle.bbox[0], vehicle.bbox[1]), (vehicle.bbox[2], vehicle.bbox[3]),(135, 107, 23),2 )
                    frame = self.putText_utf8(frame, f'{vehicle.name}', (vehicle.bbox[0],vehicle.bbox[1]),(197, 204, 100))
                    # frame = self.putText_utf8(frame, f'{vehicle.id}', (vehicle.bbox[0]-13,vehicle.bbox[1]), (255, 0, 150))
        return frame
        
    def recognition(self, frame, queue_vehicle):
        n = len(self.temporary_delete.keys())
        while n > 0:
            lst_key = list(self.temporary_delete.keys())
            per_id = lst_key[n-1]
            per = self.temporary_delete[per_id]
            if time.time() - per.delete_moment > 20:
                print(f'[Post Event:{per.name} : {strftime("%Y-%m-%d %H:%M:%S", localtime(per.last_time))}]')
                self.temporary_delete.pop(per_id)
            n -= 1
        
        frame = self.wrapper(queue_vehicle, frame)
        return frame