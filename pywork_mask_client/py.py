# -*- coding: utf-8 -*-

import copy
import time
import cv2 as cv
import threading
import time
from ffunction import *
from config import *

#初始化变量
recdic = {"LastTime":0.0,"StartTime":time.time(),"EndTime":0.0,"yobikaisu":0,"latitude":0.0,"longitude":0.0,"Total":0}
sysdic = {"time_start":0.0,"blink_counter":0,"frame_counter":0,"blink_counter_temp":0,"hclientrunning":False,"callmanrunning":False,"Low_radio_constant":Low_radio_constant,"Radio":Radio,"mean_Ratio":0,"chuuichukaisu":chuuichukaisu,"chuuibyou":chuuibyou,"himachukaisu":himachukaisu}
xydic= {"xmin":0,"ymin":0,"xmax":0,"ymax":0}














def main():
    """
    主函数
    """
    cap0 = cv.VideoCapture(0)
    cap0.set(3,Camera0Length) #摄像头分辨率
    cap0.set(4,Camera0Width)
    if disablertmp == False:
        rtmpstream = rtmppush(command)
    if disablelltcp == False:
        tcpob=valuestransferovertcp(tcpip_port)
    if disablehttp == False:
        httpob=valuetransferoverhttp()
    if disablefaced == False:
        mask=maskmodel()
        fatiguredetectobject = fatiguredetect(detector,predictor)

    while cap0.isOpened():
        while True:
            ret, frame = cap0.read()  # 读取每一帧
            frame = cv.flip(frame, 1)
            if ret:
                if disablertmp == False:
                    fframe = copy.deepcopy(frame)
                    rtmp1 = threading.Thread(target= rtmpstream.push_frame, args=[
                        fframe, sysdic, recdic])
                    rtmp1.start()
                if disableimshow == False:
                    try:
                        if disablefaced == False:
                            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                            mask.start(frame,xydic)
                            if sysdic['callmanrunning'] == False:
                                fatiguredetectobject.start(gray,recdic,sysdic,xydic)
                            img_GaussianBlur2 = cv.GaussianBlur(frame[xydic['ymin']:xydic['ymax'], xydic['xmin']:xydic['xmax']], (0, 0), 15)
                            frame[xydic['ymin']:xydic['ymax'], xydic['xmin']:xydic['xmax']] = img_GaussianBlur2
                    except Exception as e:
                        print(e)
                        pass
                    cv.imshow ("metojirukenchi", frame)
                    key = cv.waitKey(1)
                    if key == ord('q'):
                        cv.capture.release()
                        cv.destroyAllWindows()
                        exit(1)
                else:
                    if disablefaced == False and sysdic['callmanrunning'] == False:
                        try:
                            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                            mask.start(frame,xydic)
                        except Exception as e:
                            print(e)
                        fatiguredetectobject.start(gray,recdic,sysdic)
                sysdic['EndTime'] = time.time()
                try:
                    if disablelltcp == False:
                        threading.Thread(target=tcpob.sclient,args=[recdic,tcpip_port]).start()
                    if disablehttp == False:
                        threading.Thread(target=httpob.hclient,args=[recdic,httpurl,sysdic]).start()
                except Exception as e:
                    print(e)
                    pass
    cap0.release()
    cv.destroyAllWindows()
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  
    main()
