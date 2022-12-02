## config.py ##
# -*- coding: utf-8 -*-

import dlib


rtmp = 'rtmp://10.39.39.8/live/test'
httpurl = "https://baidu.com" #http接口
tcpip_port=('baidu.com',8000) #tcpsocket接口

Radio = 0.2  # 横纵比阈值
Low_radio_constant = 2  # 当Radio小于阈值时，接连多少帧一定发生眨眼动作

detector = dlib.get_frontal_face_detector() #检测器
predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')

himachukaisu = 20 #闲置眨眼次数
chuuichukaisu = 4 #眨眼次数
chuuibyou = 10 #限定秒

disableimshow=False #是否关闭本地预览
disablelltcp=True #关闭TCP Socket传输
disablehttp=True #关闭http传输
disablertmp=True #关闭rtmp
disablefaced=False
sizeStr='1280x720' #ffmepg分辨率

command = ['ffmpeg',
           '-y',
           '-r','10',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', sizeStr,
           '-i', '-',
           '-c:v', 'libx264',
           '-preset', 'faster',
           '-f', 'flv',
        rtmp] #ffmepg启动命令

Camera0Length = 1280
Camera0Width = 720