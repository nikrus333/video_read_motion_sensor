import cv2 as cv
import numpy as np
import math
from dataclasses import dataclass
import matplotlib.pyplot as plt  # $ pip install matplotlib
import matplotlib.animation as animation
import threading
import time 
from queue import Queue
import json 
#import asyncio

class CalculateCam(threading.Thread):
    def __init__(self, *args, **kwargs):
        threading.Thread.__init__(self)
        print(time.strftime('%X'))
        self.id_camera, self.mask_param, self.cropped_image_big_param, self.center_arrow, self.init_value = [args[0][key] for key in args[0]]
      

        self.past_estimate = [0, 0]  #[cost, chetvert]
        self.sepresent_estimate = [0, 0]
        self.init_value = 0
        
        self.first_step = True
        self.final_score = 0
        self.sum_score = 0
        self.cap = cv.VideoCapture(self.id_camera) 
        if not self.cap.isOpened():
            print("Cannot open camera capture")
            exit()
        else: 
            print("cam id {} openned".format(self.id_camera))
        self.cap.set(3,640)
        self.cap.set(4,480)

    def mask_black(self, img, mask):
    	#hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    	# define range of yellow color in HSV
        lower_yellow = np.array(mask[0])  # 0,100,100
        upper_yellow = np.array(mask[1])  # 30,255,255
    	# Threshold the HSV image to get only yellow colors
        mask = cv.inRange(img, lower_yellow, upper_yellow)
        return mask
    
    
    def run(self):
        flag_show = True
        while True:
            ret, frame = self.cap.read()
            frame = cv.resize(frame, (640, 480))
            if flag_show:
                cv.imshow('frame ' + str(self.id_camera), frame)

            if not ret:
                print("Can't receive frame, exiting")
            
            cropped_image_big = frame[self.cropped_image_big_param[0]:self.cropped_image_big_param[1],
                                      self.cropped_image_big_param[2]:self.cropped_image_big_param[3]]#[140:340, 185:390]
            cropped_image_small = frame[169:230, 310:390] # y x
            #cv.imshow('cropped_image_small', cropped_image_small)
            if flag_show:
                cv.imshow('cropped_image_big ' + str(self.id_camera), cropped_image_big)
            
            #cropped_image_big_show = np.copy(frame[190:295, 250:380])
            cropped_image_small_bin = self.mask_black(cropped_image_small, mask=self.mask_param)
            cropped_image_big_bin = self.mask_black(cropped_image_big, mask=self.mask_param)
            #cv.imshow('cropped_image_small_bin', cropped_image_small_bin)
            if flag_show:
                cv.imshow('cropped_image_big_bin '+ str(self.id_camera), cropped_image_big_bin)
            
            lines_small = cv.HoughLinesP(cropped_image_small_bin,1,np.pi/180,40,minLineLength=20,maxLineGap=30)
            lines_big = cv.HoughLinesP(cropped_image_big_bin,1,np.pi/180,40,minLineLength=40,maxLineGap=40)
            
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
                    cv.line(cropped_image_big,(x1,y1),(x2,y2),(255,0,0),5)
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
                        print("Микрометры устройтсво {0} ИТОГ : {1}".format(self.id_camera, absolut + self.final_score)) 
                        self.sum_score = absolut + self.final_score
                        #self.queue1.put({self.id_camera : absolut + self.final_score})
            if flag_show:
                cv.imshow("line_big " + str(self.id_camera),cropped_image_big)
            if cv.waitKey(1) == ord('q'):
                self.cap.release()
                cv.destroyAllWindows()
     
class GetValue():
    def __init__(self):
        f = open('data_prarams.json', 'r')
        data_param = json.loads(f.read())
        cam_0 = data_param['Camera0']
        cam_1 = data_param['Camera1']
        cam_2 = data_param['Camera2']
        f.close()
        # self.vid = CalculateCam(cam_0)
        # self.vid.start()
        self.vid1 = CalculateCam(cam_1)
        self.vid1.start()
        self.vid2 = CalculateCam(cam_2)
        self.vid2.start()
        
        
    def value(self):
        return(self.vid.sum_score, self.vid1.sum_score, self.vid2.sum_score)
        
def main():
    get = GetValue()
    print('start')
    # while True:
    #     print(get.value())
    # # print('start1')
    #     time.sleep(5)
    # print('start2')
    # print(get.value())
    # time.sleep(5)
    # print('start3')
    # print(get.value())



if __name__ == '__main__':
    main()
    


# open cv - video capture +
# binarization - 1 bit image +
# optional - crop
# line detection
# filter array of lines by length
# get the largest of the two
# determine angle
# convert angle -> length
# print to file timestamp - length
# print image to file w timestamp
