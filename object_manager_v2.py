import time
from face_models.tritonExtractor import TritonExtractor, compute_sim
import glob

import os
import cv2
from face_models.face_align import norm_crop
from PIL import ImageFont, ImageDraw, Image

import numpy as np


class Person():
    def __init__(self, id):
        self.id = id
        
        self.face_id = None
        
        self.name = "stranger"
        self.bbox = []

        self.first_time = 0
        self.last_time = 0
        
        self.landmark = None
        self.face_image = None
        self.identied = False
        
class PeopleManager():
    def __init__(self):
        self.list_people = []
        self.dict_people = {}
        self.recogized_people_dict = {}

        self.current_time = time.time()
        self.emb_lst = []
        self.name_lst = []
        self.extractor = TritonExtractor()

        self.font = ImageFont.truetype("font/RobotoSlab-Regular.ttf", 20)

        lst_emb = glob.glob('/face_service/emb_test/*')
        for emb_np in lst_emb:
            name = os.path.basename(emb_np).split('.')[0]
            self.emb_lst.append(np.load(emb_np))
            self.name_lst.append(name)
        

    def add_person(self, person):
        self.list_people.append(person)
        self.dict_people.update({str(person.id): person})

    # @measure_time
    def update_tracking(self, tracking_results, frame):

        self.current_time = time.time()
        # self.current_datetime = datetime.datetime.now()
        for track in tracking_results:
            
            # track_bbox = track[0]
            track_bbox = np.array(track[:4]).astype(np.int32)
            x1,y1,x2,y2 = track_bbox
            track_id = track[4]
            track_lmk = track[5]
            person_id = str(track_id)
            # print(track_lmk)

            # cropped = norm_crop(frame, track_lmk)
            # # cv2.imwrite('face_check.jpg', cropped)
            # scores = []
            # emb1 = self.extractor.get_embedding(cropped)
            # for emb2 in self.emb_lst:
            #     score = compute_sim(emb1, emb2)
            #     scores.append(score)

            #     # print('scores:',scores)
            # if len(scores):
            #     max_scores_id = np.argmax(scores)
            #     # print("max_scores_id" ,max_scores_id)
            #     # print('max scores:',scores[max_scores_id])
            #     name = 'stranger'
            #     if scores[max_scores_id] > 0.55:
            #         name = self.name_lst[max_scores_id]
            #         print(name)

            #     cv2.rectangle(frame,(track_bbox[0], track_bbox[1]), (track_bbox[2], track_bbox[3]), (255, 128, 128),1 )
            #     frame = self.putText_utf8(frame, f'{name}', (track_bbox[0],track_bbox[1]), (255, 207, 150))

            # update person info if person in queue
            if person_id in self.dict_people:
                person = self.dict_people[person_id]
                person.bbox = track_bbox
                person.landmark = track_lmk

                person.last_time = time.time()
                person.face_image = frame[y1:y2,x1:x2]
            # create new person info if tracking not in queue
            else:
                new_person_id = str(person_id)
                person = Person(new_person_id)
                person.bbox = track_bbox
                person.landmark = track_lmk

                person.first_time = time.time()
                person.last_time = time.time()
                
                try:
                    person.face_image = frame[y1:y2,x1:x2]
                except:
                    print('Cant label face for person')
                person.landmark = track_lmk
                ### add person 
                self.add_person(person)
            
        frame = self.recognition(frame)
        return frame
    
    def update_recognition_info(self, recog_results):
        for res in recog_results:
            pass    
    def putText_utf8(self, img, text, pos, color):
        x, y = pos
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_img)

        # Draw non-ascii text onto image
        draw = ImageDraw.Draw(pil_image)
        draw.text((x,y-25), text, font=self.font, fill=self.cvt_color_putText_BGR2RGB(color))

        # Convert back to Numpy array and switch back from RGB to BGR
        image = np.asarray(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image

    def cvt_color_putText_BGR2RGB(self, color):
        b,g,r = color
        return (r, g, b)
    
    def recognition(self, frame, draw=True):
        i = 0
        n_people = len(self.list_people)
        embs = np.empty((0,512))
        
        while n_people > 0:
            person_batch = []

            while len(person_batch) < 5:
                if n_people > 0:
                    if not self.list_people[i].identied:
                        person_batch.append(self.list_people[i])
                    i += 1
                    n_people -= 1
                else:
                    break

            input_len = len(person_batch)

            if len(person_batch):
                batch_results = self.infer_batch(frame, person_batch)[:input_len]

                embs = np.vstack([embs, batch_results])
        
        ind_emb = 0

        for person  in self.list_people:
            if not person.identied:
                scores = []
                for emb2 in self.emb_lst:
                    
                    score = compute_sim(embs[ind_emb], emb2)
                    scores.append(score)
                ind_emb += 1
                if len(scores):
                    max_scores_id = np.argmax(scores)
                    # print("max_scores_id" ,max_scores_id)

                    if scores[max_scores_id] > 0.6:
                        person.identied = True
                        person.name = self.name_lst[max_scores_id]
                        print(person.name)
                    else:
                        person.identied = False
                        person.name = 'stranger'

            if time.time() - person.last_time > 1:
                self.dict_people.pop(str(person.id))
                self.list_people.remove(person)
            
            if draw and time.time() - person.last_time < 0.1:
                if not person.identied:
                    cv2.rectangle(frame,(person.bbox[0], person.bbox[1]), (person.bbox[2], person.bbox[3]), (0, 15, 153),2 )
                    frame = self.putText_utf8(frame, f'Stranger', (person.bbox[0],person.bbox[1]), (0, 15, 153))
                else:
                    # print('dra')
                    cv2.rectangle(frame,(person.bbox[0], person.bbox[1]), (person.bbox[2], person.bbox[3]), (255, 128, 128),2 )
                    frame = self.putText_utf8(frame, f'{person.name}', (person.bbox[0],person.bbox[1]), (255, 207, 150))
                    # frame = self.putText_utf8(frame, f'{person.id}', (person.bbox[0]-13,person.bbox[1]), (255, 0, 150))

        return frame
            
    def preprocess(self, frame, person_batch, batch_size=5):
        face_imgs = []
        
        for i in range(batch_size):
            if i < len(person_batch):
                face_imgs.append(norm_crop(frame, person_batch[i].landmark))
            else:
                face_imgs.append(np.zeros((112,112,3)))

        return face_imgs

    def postprocess(self, batch_emb, person_batch):
        for i, emb in enumerate(batch_emb):
            person_batch[i].emb = emb
        pass

    def infer_batch(self, frame, person_batch):
        preprocessed_batch = self.preprocess(frame, person_batch)
        embs = self.extractor.get_embedding_batch(preprocessed_batch)
        return embs