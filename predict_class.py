
import time
from tkinter.messagebox import NO

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class UIDetecter():
    def __init__(self, mainwin):
        self.yolo = YOLO()
        self.mainwin = mainwin

    def displayImg_out(self, img):
        '''
        label_out 中显示预测结果
        '''
        img = self.mainwin.padding(img)
        RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img_out = QImage(RGBImg, RGBImg.shape[1], RGBImg.shape[0], QImage.Format_RGBA8888)
        img_out = QPixmap(img_out)
        # img_out = img_out.scaledToWidth(self.mainwin.labelsize[1])
        img_out = self.mainwin.resizeImg(img_out)

        self.mainwin.label_out.setPixmap(img_out)


    def displayImg_in(self, img):
        '''
        label_in 中显示输入图像
        '''
        img = self.mainwin.padding(img)
        RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img_out = QImage(RGBImg, RGBImg.shape[1], RGBImg.shape[0], QImage.Format_RGBA8888)
        img_out = QPixmap(img_out)
        # img_out = img_out.scaledToWidth(self.mainwin.labelsize[1])
        img_out = self.mainwin.resizeImg(img_out)

        self.mainwin.label_in.setPixmap(img_out)

    def display_result(self, results):
        if results is None:
            return
        self.mainwin.listWidget_result.clear()
        for result in results:
            if result[1] is not None:
                item = result[0] + ': ' + result[1]
                self.mainwin.listWidget_result.addItem(item)
            

    def run(self, mode, source, save_path, classes):
        if mode == 'image':
            try:
                image = Image.open(source)
                img_in = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
                self.displayImg_in(img_in)
            except:
                print('Open Error! Try again!')
            else:
                im, result = self.yolo.detect_image(image, classes, crop = False, count=False)
                r_image = np.array(im)
                img_out = cv2.cvtColor(r_image,cv2.COLOR_RGB2BGR)
                self.displayImg_out(img_out)
                self.display_result(result)
                
        elif mode == 'video':
            cap = []
            video_path = source
            for i in range(len(video_path)):
                cap.append(cv2.VideoCapture(video_path[i]))

            width = (int(cap[0].get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = (int(cap[0].get(cv2.CAP_PROP_FRAME_HEIGHT)))

            if save_path!="":
                fourcc  = cv2.VideoWriter_fourcc(*'XVID')
                size    = (int(cap[0].get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap[0].get(cv2.CAP_PROP_FRAME_HEIGHT)))
                out     = cv2.VideoWriter(save_path, fourcc, 25, size)

            for i in range(len(video_path)):
                ref, frame = cap[i].read()
                if not ref:
                    raise ValueError("未能正确读取摄像头1（视频1），请注意是否正确安装摄像头（是否正确填写视频路径）。")

            fps = 0.0
            while(True):
                t1 = time.time()
                # 读取某一帧
                frame_list = []
                for i in range(len(video_path)):
                    ref, frame = cap[i].read()
                    if not ref:
                        break
                    frame_list.append(cv2.resize(frame, (int(width/2), int(height/2)), interpolation=cv2.INTER_CUBIC))
                if not ref:
                    break
                if len(frame_list) == 2:
                    frameUp = np.hstack((frame_list[0], frame_list[1]))
                elif len(frame_list) == 1:
                    frameUp = frame_list[0]
                self.displayImg_in(frameUp)

                # 格式转变，BGRtoRGB
                frame = cv2.cvtColor(frameUp,cv2.COLOR_BGR2RGB)
                # 转变成Image
                frame = Image.fromarray(np.uint8(frame))
                # 进行检测
                im, result = self.yolo.detect_image(frame, classes)
                frame = np.array(im)
                # RGBtoBGR满足opencv显示格式
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

                fps  = ( fps + (1./(time.time()-t1)) ) / 2
                # print("fps= %.2f"%(fps))
                frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # cv2.imshow("video",frame)
                # c= cv2.waitKey(1) & 0xff
                self.displayImg_out(frame)
                self.display_result(result)
                if save_path!="":
                    out.write(frame)

                if self.mainwin.stop == 1:
                    for i in range(len(video_path)):
                        cap[i].release()
                    break

            print("Video Detection Done!")
            for i in range(len(video_path)):
                cap[i].release()
            if save_path!="":
                print("Save processed video to the path :" + save_path)
                out.release()
            cv2.destroyAllWindows()
