import json
from threading import Thread, Lock
import cv2
import numpy as np
import math
import time
import pandas as pd

class WebcamVideoStream:
    def __init__(self, *args ) :
        self.id_camera, self.mask_param, self.cropped_image_big_param, self.center_arrow, self.init_value = [args[0][key] for key in args[0]]
        self.stream = cv2.VideoCapture(self.id_camera)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()
        self.past_estimate = [0, 0]  #[cost, chetvert]
        self.sepresent_estimate = [0, 0]
        self.init_value = 0
        
        self.first_step = True
        self.final_score = 0
        self.sum_score = 0

    def start(self):
        if self.started :
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self):
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def mask_black(self, img, mask):
    	#hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    	# define range of yellow color in HSV
        lower_yellow = np.array(mask[0])  # 0,100,100
        upper_yellow = np.array(mask[1])  # 30,255,255
    	# Threshold the HSV image to get only yellow colors
        mask = cv2.inRange(img, lower_yellow, upper_yellow)
        return mask
    
    def work(self, frame):
        frame = cv2.resize(frame, (640, 480))
   
            
        cropped_image_big = frame[self.cropped_image_big_param[0]:self.cropped_image_big_param[1],
                                  self.cropped_image_big_param[2]:self.cropped_image_big_param[3]]#[140:340, 185:390]
        cropped_image_small = frame[169:230, 310:390] # y x
        #cv.imshow('cropped_image_small', cropped_image_small)

        
        #cropped_image_big_show = np.copy(frame[190:295, 250:380])
        cropped_image_small_bin = self.mask_black(cropped_image_small, mask=self.mask_param)
        cropped_image_big_bin = self.mask_black(cropped_image_big, mask=self.mask_param)
        #cv.imshow('cropped_image_small_bin', cropped_image_small_bin)
      
        
        lines_small = cv2.HoughLinesP(cropped_image_small_bin,1,np.pi/180,40,minLineLength=20,maxLineGap=30)
        lines_big = cv2.HoughLinesP(cropped_image_big_bin,1,np.pi/180,40,minLineLength=40,maxLineGap=40)
        
        if lines_small is None:
            pass
        else:
            i = 0
            for x1,y1,x2,y2 in lines_small[0]:
                i+=1
                #cv.line(cropped_image_small,(x1,y1),(x2,y2),(255,0,0),5)
                #alpha = math.atan ( (y2-y1) / (x2-x1) )
                #alpha_degree = math.degrees(alpha)
                #print("угол наклона прямой маленькой стрелки {0}".format(alpha_degree))
        center_arrow = self.center_arrow#[115, 86] # [x , y]
        if lines_big is None:
            pass
        else:
            i = 0
            for x1,y1,x2,y2 in lines_big[0]:
                i+=1
                cv2.line(cropped_image_big,(x1,y1),(x2,y2),(255,0,0),5)
                alpha = math.atan( (y2-y1) / (x2-x1) )
                alpha = math.atan2( (y2-y1) , (x2-x1) )
                alpha_degree = math.degrees(alpha)
                #print("угол наклона прямой большой стрелки {0}".format(alpha_degree))
                distance_point_1 = math.sqrt((x1 - center_arrow[0])**2 + (y1 - center_arrow[1])**2)
                distance_point_2 = math.sqrt((x2 - center_arrow[0])**2 + (y2 - center_arrow[1])**2)
                #select point
                if distance_point_1 >= distance_point_2:
                    goal_point = [x1, y1]
                else:
                    goal_point = [x2, y2]
                
                #select region axes
                if goal_point[0] > center_arrow[0]:
                    if goal_point[1] > center_arrow[1]:
                        chetvert = 4
                    else:
                        chetvert = 1
                    angle_final = 90 + alpha_degree
                else:
                    if goal_point[1] > center_arrow[1]:
                        chetvert = 3
                    else:
                        chetvert = 2
                    angle_final = 270 + alpha_degree
                    #print("четвреть где находится конец стрелки {0}".format(chetvert))   
                #print("угол наклона прямой большой стрелки {0}".format(angle_final))   
                step = 100 / 360
                absolut = step * angle_final
                #print("Микрометры {0}".format(absolut))  
                #filter
                #calcualte
                if not self.first_step:
                    self.past_estimate = [absolut, chetvert]
                    self.present_estimate = [absolut, chetvert]
                    self.first_step = False
                else: 
                    self.present_estimate = [absolut, chetvert]
                    if self.present_estimate[1] == 1 and self.past_estimate[1] == 2:
                        self.final_score +=100
                    if self.present_estimate[1] == 2 and self.past_estimate[1] == 1:
                        self.final_score -=100
                    if self.present_estimate[1] == 2 and self.past_estimate[1] == 4:
                        continue
                    if self.present_estimate[1] == 4 and self.past_estimate[1] == 2:
                        continue
                    if self.present_estimate[1] == 1 and self.past_estimate[1] == 3:
                        continue
                    if self.present_estimate[1] == 3 and self.past_estimate[1] == 1:
                        continue
                    self.past_estimate = [absolut, chetvert]
                    #print("Микрометры устройтсво {0} ИТОГ : {1}".format(self.id_camera, absolut + self.final_score)) 
                    self.sum_score = absolut + self.final_score
                    #self.queue1.put({self.id_camera : absolut + self.final_score})
        
        if cv2.waitKey(1) == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
        return cropped_image_big, cropped_image_big_bin, self.sum_score, 

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()

class GetValue():
    def __init__(self):
        f = open('data_prarams.json', 'r')
        data_param = json.loads(f.read())
        cam_0 = data_param['Camera0']
        cam_1 = data_param['Camera1']
        cam_2 = data_param['Camera2']

    
        f.close()
        self.vs = WebcamVideoStream(cam_0).start()
        self.vs1 = WebcamVideoStream(cam_1).start()
        self.vs2 = WebcamVideoStream(cam_2).start()

        self.value0 = 0
        self.value1 = 0
        self.value2 = 0
    
    def nothing(sekf, x):
        pass
        

    def work(self):
        while True :
            frame = self.vs.read()
            frame1 = self.vs1.read()
            frame2 = self.vs2.read()
            cropped_image_big, bin_image, self.value0 = self.vs.work(frame)
            cropped_image_big1, bin_image1, self.value1 = self.vs1.work(frame1)
            cropped_image_big2, bin_image2, self.value2 = self.vs2.work(frame2)
            cv2.imshow('webcam', frame)
            cv2.imshow('webcam1', frame1)
            cv2.imshow('webcam2', frame2)
            cv2.imshow('cropped_image_big', cropped_image_big)
            cv2.imshow('cropped_image_big1', cropped_image_big1)
            cv2.imshow('cropped_image_big2', cropped_image_big2)
            cv2.imshow('image_bin', bin_image)
            cv2.imshow('image_bin1', bin_image1)
            cv2.imshow('image_bin2', bin_image2)
            if cv2.waitKey(1) == 27 :
                break
        self.vs.stop()
        self.vs1.stop()
        self.vs2.stop()
        cv2.destroyAllWindows()

    def value(self):
        return[self.value0, self.value1, self.value2]
    

if __name__ == "__main__" :
    getvalue = GetValue()
    x = Thread(target= getvalue.work)
    x.start()
    x_axes_data = []
    y_axes_data = []
    z_axes_data = []
    try:
        while True:
            value = getvalue.value()
            print(value)
            x_axes_data.append(value[0])
            y_axes_data.append(value[1])
            z_axes_data.append(value[2])
            time.sleep(0.1)
            
    except KeyboardInterrupt:
            data = dict(x_axes_data=x_axes_data, y_axes_data=y_axes_data, z_axes_data=z_axes_data)
            df = pd.DataFrame(data)
            df.to_csv('value_axises__2.csv')
            print('end work')
    
