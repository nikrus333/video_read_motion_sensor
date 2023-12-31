import cv2 as cv
import numpy as np
import math
from dataclasses import dataclass
import matplotlib.pyplot as plt  # $ pip install matplotlib
import matplotlib.animation as animation
import threading
import time 
from queue import Queue
#import asyncio

class CalculateCam(threading.Thread):
    def __init__(self, queue1,  cam_nomber : int = 0, ) -> None:
        threading.Thread.__init__(self)
        print(time.strftime('%X'))
        self.queue1 = queue1
        self.past_estimate = [0, 0]  #[cost, chetvert]
        self.sepresent_estimate = [0, 0]
        self.id_camera = cam_nomber
        self.first_step = True
        self.final_score = 0
        self.sum_score = 0
        self.cap = cv.VideoCapture(cam_nomber)
        if not self.cap.isOpened():
            print("Cannot open camera capture")
            exit()

    def mask_black(self, img):
    	#hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    	# define range of yellow color in HSV
        lower_yellow = np.array([4, 4, 4])  # 0,100,100
        upper_yellow = np.array([60, 60, 70])  # 30,255,255
    	# Threshold the HSV image to get only yellow colors
        mask = cv.inRange(img, lower_yellow, upper_yellow)
        return mask
    
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            frame = cv.resize(frame, (640, 480))
            cv.imshow('frame ' + str(self.id_camera), frame)

            if not ret:
                print("Can't receive frame, exiting")
            
            cropped_image_big = frame[140:340, 185:390]
            cropped_image_small = frame[169:230, 310:390] # y x
            #cv.imshow('cropped_image_small', cropped_image_small)
            cv.imshow('cropped_image_big ' + str(self.id_camera), cropped_image_big)
            
            #cropped_image_big_show = np.copy(frame[190:295, 250:380])
            cropped_image_small_bin = self.mask_black(cropped_image_small)
            cropped_image_big_bin = self.mask_black(cropped_image_big)
            #cv.imshow('cropped_image_small_bin', cropped_image_small_bin)
            cv.imshow('cropped_image_big_bin '+ str(self.id_camera), cropped_image_big_bin)
            
            lines_small = cv.HoughLinesP(cropped_image_small_bin,1,np.pi/180,40,minLineLength=20,maxLineGap=30)
            lines_big = cv.HoughLinesP(cropped_image_big_bin,1,np.pi/180,40,minLineLength=20,maxLineGap=30)
            
            if lines_small is None:
                pass
            else:
                i = 0
                for x1,y1,x2,y2 in lines_small[0]:
                    i+=1
                    cv.line(cropped_image_small,(x1,y1),(x2,y2),(255,0,0),5)
                    alpha = math.atan ( (y2-y1) / (x2-x1) )
                    alpha_degree = math.degrees(alpha)
                    #print("угол наклона прямой маленькой стрелки {0}".format(alpha_degree))

            center_arrow = [115, 86] # [x , y]
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
                        self.queue1.put({self.id_camera : absolut + self.final_score})
            if cv.waitKey(1) == ord('q'):
                
                self.cap.release()
                cv.destroyAllWindows()
     
class GetValue(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        queue1 = Queue()
        self.vid = CalculateCam(queue1, 2)
        self.vid.start()
        self.vid1 = CalculateCam(queue1, 0)
        self.vid1.start()
    def value(self):
        return(self.vid.sum_score, self.vid1.sum_score)
        
def main():
    get = GetValue()
    print('start')
    
    print(get.value())
    print('start1')
    time.sleep(5)
    print('start2')
    print(get.value())
    time.sleep(5)
    print(get.value())

    #get.start()
    
    # vid = CalculateCam(2)
    # vid.start()
    # #print(vid.sum_score)
    # vid1 = CalculateCam(0)
    # vid1.start()
    # # while True:
    #     print('vid1 ',vid1.final_score,'  vid  ', vid.final_score)


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
