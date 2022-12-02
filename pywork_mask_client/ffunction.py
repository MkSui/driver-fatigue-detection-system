# -*- coding: utf-8 -*-

import base64
import hashlib
import multiprocessing
import socket
import subprocess
import time
import requests
import json
import cv2 as cv
import pygame
import dlib
import numpy as np
from scipy.spatial import distance
import threading
from pytorch_loader import load_pytorch_model, pytorch_inference
import sys
import torch


class maskmodel(object):
    def start(self,frame,xydic):
        print(xydic)
        multiprocessing.Process(target=self.inference,args=[frame,xydic]).start()

    def __init__(self):
        sys.path.append('models/')

    def load_pytorch_model(self,model_path):
        model = torch.load(model_path)
        return model

    def pytorch_inference(self,model, img_arr):
        if torch.cuda.is_available():
            dev = 'cuda:0'
        else:
            dev = 'cpu'
        device = torch.device(dev)
        model.to(device)
        input_tensor = torch.tensor(img_arr).float().to(device)
        y_bboxes, y_scores, = model.forward(input_tensor)
        return y_bboxes.detach().cpu().numpy(), y_scores.detach().cpu().numpy()


    def single_class_non_max_suppression(self,bboxes, confidences, conf_thresh=0.2, iou_thresh=0.5, keep_top_k=-1):
        '''
        do nms on single class.
        Hint: for the specific class, given the bbox and its confidence,
        1) sort the bbox according to the confidence from top to down, we call this a set
        2) select the bbox with the highest confidence, remove it from set, and do IOU calculate with the rest bbox
        3) remove the bbox whose IOU is higher than the iou_thresh from the set,
        4) loop step 2 and 3, util the set is empty.
        :param bboxes: numpy array of 2D, [num_bboxes, 4]
        :param confidences: numpy array of 1D. [num_bboxes]
        :param conf_thresh:
        :param iou_thresh:
        :param keep_top_k:
        :return:
        '''
        if len(bboxes) == 0: return []

        conf_keep_idx = np.where(confidences > conf_thresh)[0]

        bboxes = bboxes[conf_keep_idx]
        confidences = confidences[conf_keep_idx]

        pick = []
        xmin = bboxes[:, 0]
        ymin = bboxes[:, 1]
        xmax = bboxes[:, 2]
        ymax = bboxes[:, 3]

        area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)
        idxs = np.argsort(confidences)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # keep top k
            if keep_top_k != -1:
                if len(pick) >= keep_top_k:
                    break

            overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
            overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
            overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
            overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
            overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
            overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
            overlap_area = overlap_w * overlap_h
            overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)

            need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
            idxs = np.delete(idxs, need_to_be_deleted_idx)

        # if the number of final bboxes is less than keep_top_k, we need to pad it.
        # TODO
        return conf_keep_idx[pick]


    def decode_bbox(self,anchors, raw_outputs, variances=[0.1, 0.1, 0.2, 0.2]):
        '''
        Decode the actual bbox according to the anchors.
        the anchor value order is:[xmin,ymin, xmax, ymax]
        :param anchors: numpy array with shape [batch, num_anchors, 4]
        :param raw_outputs: numpy array with the same shape with anchors
        :param variances: list of float, default=[0.1, 0.1, 0.2, 0.2]
        :return:
        '''
        anchor_centers_x = (anchors[:, :, 0:1] + anchors[:, :, 2:3]) / 2
        anchor_centers_y = (anchors[:, :, 1:2] + anchors[:, :, 3:]) / 2
        anchors_w = anchors[:, :, 2:3] - anchors[:, :, 0:1]
        anchors_h = anchors[:, :, 3:] - anchors[:, :, 1:2]
        raw_outputs_rescale = raw_outputs * np.array(variances)
        predict_center_x = raw_outputs_rescale[:, :, 0:1] * anchors_w + anchor_centers_x
        predict_center_y = raw_outputs_rescale[:, :, 1:2] * anchors_h + anchor_centers_y
        predict_w = np.exp(raw_outputs_rescale[:, :, 2:3]) * anchors_w
        predict_h = np.exp(raw_outputs_rescale[:, :, 3:]) * anchors_h
        predict_xmin = predict_center_x - predict_w / 2
        predict_ymin = predict_center_y - predict_h / 2
        predict_xmax = predict_center_x + predict_w / 2
        predict_ymax = predict_center_y + predict_h / 2
        predict_bbox = np.concatenate([predict_xmin, predict_ymin, predict_xmax, predict_ymax], axis=-1)
        return predict_bbox
        
    def generate_anchors(self,feature_map_sizes, anchor_sizes, anchor_ratios, offset=0.5):
        '''
        generate anchors.
        :param feature_map_sizes: list of list, for example: [[40,40], [20,20]]
        :param anchor_sizes: list of list, for example: [[0.05, 0.075], [0.1, 0.15]]
        :param anchor_ratios: list of list, for example: [[1, 0.5], [1, 0.5]]
        :param offset: default to 0.5
        :return:
        '''
        anchor_bboxes = []
        for idx, feature_size in enumerate(feature_map_sizes):
            cx = (np.linspace(0, feature_size[0] - 1, feature_size[0]) + 0.5) / feature_size[0]
            cy = (np.linspace(0, feature_size[1] - 1, feature_size[1]) + 0.5) / feature_size[1]
            cx_grid, cy_grid = np.meshgrid(cx, cy)
            cx_grid_expend = np.expand_dims(cx_grid, axis=-1)
            cy_grid_expend = np.expand_dims(cy_grid, axis=-1)
            center = np.concatenate((cx_grid_expend, cy_grid_expend), axis=-1)

            num_anchors = len(anchor_sizes[idx]) +  len(anchor_ratios[idx]) - 1
            center_tiled = np.tile(center, (1, 1, 2* num_anchors))
            anchor_width_heights = []

            # different scales with the first aspect ratio
            for scale in anchor_sizes[idx]:
                ratio = anchor_ratios[idx][0] # select the first ratio
                width = scale * np.sqrt(ratio)
                height = scale / np.sqrt(ratio)
                anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

            # the first scale, with different aspect ratios (except the first one)
            for ratio in anchor_ratios[idx][1:]:
                s1 = anchor_sizes[idx][0] # select the first scale
                width = s1 * np.sqrt(ratio)
                height = s1 / np.sqrt(ratio)
                anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

            bbox_coords = center_tiled + np.array(anchor_width_heights)
            bbox_coords_reshape = bbox_coords.reshape((-1, 4))
            anchor_bboxes.append(bbox_coords_reshape)
        anchor_bboxes = np.concatenate(anchor_bboxes, axis=0)
        return anchor_bboxes
        
    def inference(self,image,xydic):
        '''
        Main function of detection inference
        :param image: 3D numpy array of image
        :param conf_thresh: the min threshold of classification probabity.
        :param iou_thresh: the IOU threshold of NMS
        :param target_shape: the model input size.
        :param draw_result: whether to daw bounding box to the image.
        :param show_result: whether to display the image.
        :return:
        '''
                # model = load_pytorch_model('models/face_mask_detection.pth');
        conf_thresh=0.5
        iou_thresh=0.5
        target_shape=(360, 360)
        draw_result=False
        model = self.load_pytorch_model('models/model360.pth')
        id2class = {0: 'Mask', 1: 'NoMask'}

        # anchor configuration
        #feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
        feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
        anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
        anchor_ratios = [[1, 0.62, 0.42]] * 5

        # generate anchors
        anchors = self.generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

        # for inference , the batch size is 1, the model output shape is [1, N, 4],
        # so we expand dim for anchors to [1, anchor_num, 4]
        anchors_exp = np.expand_dims(anchors, axis=0)
        # image = np.copy(image)
        output_info = []
        height, width, _ = image.shape
        image_resized = cv.resize(image, target_shape)
        image_np = image_resized / 255.0  # 归一化到0~1
        image_exp = np.expand_dims(image_np, axis=0)

        image_transposed = image_exp.transpose((0, 3, 1, 2))

        y_bboxes_output, y_cls_output = pytorch_inference(model, image_transposed)
        # remove the batch dimension, for batch is always 1 for inference.
        y_bboxes = self.decode_bbox(anchors_exp, y_bboxes_output)[0]
        y_cls = y_cls_output[0]
        # To speed up, do single class NMS, not multiple classes NMS.
        bbox_max_scores = np.max(y_cls, axis=1)
        bbox_max_score_classes = np.argmax(y_cls, axis=1)

        # keep_idx is the alive bounding box after nms.
        keep_idxs = self.single_class_non_max_suppression(y_bboxes,
                                                    bbox_max_scores,
                                                    conf_thresh=conf_thresh,
                                                    iou_thresh=iou_thresh,
                                                    )

        for idx in keep_idxs:
            conf = float(bbox_max_scores[idx])
            class_id = bbox_max_score_classes[idx]
            bbox = y_bboxes[idx]
            # clip the coordinate, avoid the value exceed the image boundary.
            xmin = max(0, int(bbox[0] * width))
            ymin = max(0, int(bbox[1] * height))
            xmax = min(int(bbox[2] * width), width)
            ymax = min(int(bbox[3] * height), height)


        xydic['xmin']=xmin
        xydic['ymin']=ymin
        xydic['xmax']=xmax
        xydic['ymax']=ymax
        print(xydic)
        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv.FONT_HERSHEY_SIMPLEX, 0.8, color)
            img_GaussianBlur2 = cv.GaussianBlur(image[ymin:ymax, xmin:xmax], (0, 0), 15)
            image[ymin:ymax, xmin:xmax] = img_GaussianBlur2
    
class valuestransferovertcp(object): #tcp流传输
    def __init__(self,tcpip_port):
        client=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        client.connect(tcpip_port)
    def sclient(self,datalist):
        json_string = json.dumps(datalist)
        ascii_message = json_string.encode('ascii')
        message=base64.b64encode(ascii_message)
        self.client.send(message)#将发送的数据进行编码

class valuetransferoverhttp(object): #http流传输
    def hclient(self,datalist,url,sysdic):
        if sysdic['hclientrunning'] == False:
            sysdic['hclientrunning'] = True 
            json_string = json.dumps(datalist)
            #print(json_string)
            key=time.strftime("%m-%d-%H", time.localtime())
            hash5 = hashlib.md5()
            hash5.update(key.encode("utf-8"))
            params = {
            "loginkey": (hash5.hexdigest()).upper(),
            "jsondata": json_string.encode('utf-8')
        }
            requests.get(url,params=params)
            time.sleep(2)
            sysdic['hclientrunning'] = False

class rtmppush(object): #rtmp/hls
    def __init__(self,command):
        self.command = command
        self.ffpipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)

    def __frame_handle__(self,frame,sysdic,recdic):
    # 显示结果
        cv.putText(frame, "{}".format(time.strftime("%m-%d %H:%M:%S", time.localtime())), (10, 30),
                cv.FONT_HERSHEY_SIMPLEX, 0.7, [255, 255, 255], 2)
        cv.putText(frame, "Ratio{:.2f}".format(sysdic['mean_Ratio']), (620, 30),
                cv.FONT_HERSHEY_SIMPLEX, 0.7, [34,155,175], 2)
        cv.putText(frame, str(recdic['latitude'])+"   "+str(recdic['longitude']), (1150, 30),
                cv.FONT_HERSHEY_SIMPLEX, 0.7, [152,236,15], 2)
        if recdic['yobikaisu'] > 2:
            cv.putText(frame, "Level {}".format(recdic['yobikaisu']), (10, 710),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, [0, 0, 255], 2)
        else:
            cv.putText(frame, "Level {}".format(recdic['yobikaisu']), (10, 710),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, [255, 255, 255], 2) 
        #输入到ffmpeg流
        return frame
    def push_frame(self,frame, sysdic, recdic):
                # 对获取的帧进行画面处理
                nframe = self.__frame_handle__(frame, sysdic, recdic)
                try:
                    self.ffpipe.stdin.write(nframe.tobytes())
                except IOError:
                    print("Rebuild ffmpeg process")
                    self.ffpipe = subprocess.Popen(self.command, shell=False, stdin=subprocess.PIPE ,stderr=subprocess.STDOUT,stdout=subprocess.DEVNULL)
                except Exception as e:
                    print(e)


class fatiguredetect(object):
    def start(self,gray,recdic,sysdic,xydic):
        multiprocessing.Process(target=self.eyedetect,args=[gray,recdic,sysdic,xydic]).start()
    def __init__(self,detector,predictor):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')
        self.LeftEye_Start = 36
        self.LeftEye_End = 41
        self.RightEye_Start = 42
        self.RightEye_End = 47
        file = 'warning.mp3'
        #pygame.mixer.init()
        #pygame.mixer.music.load(file)
    def callman(self,yobikaisu,recdic,sysdic):
        sysdic['callmanrunning'] = True
        print("警报！！！请保持清醒状态驾驶")
        #pygame.mixer.music.play(yobikaisu)
        recdic['LastTime']=time.time()
        recdic['Total'] += 1
        time.sleep(6)
        sysdic['callmanrunning'] = False



    def eyedetect(self,gray,recdic,sysdic,xydic):
        rect=dlib.rectangle(int(xydic['xmin']), int(xydic['ymin']), int(xydic['xmax']), int(xydic['ymax']))
        rects=dlib.rectangles()
        rects.append(rect)
        # 设定人眼标定点
        for rect in rects:
            shape = self.predictor(gray, rect)
            points = np.zeros((68, 2), dtype=int)
            for i in range(68):
                points[i] = (shape.part(i).x, shape.part(i).y)

            # 获取眼睛特征点
            Lefteye = points[self.LeftEye_Start: self.LeftEye_End + 1]
            Righteye = points[self.RightEye_Start: self.RightEye_End + 1]

            # 计算眼睛横纵比
            Lefteye_Ratio = self.__calculate_Ratio(Lefteye)
            Righteye_Ratio = self.__calculate_Ratio(Righteye)
            sysdic['mean_Ratio'] = (Lefteye_Ratio + Righteye_Ratio) / 2  # 计算两眼平均比例

            # 计算凸包
            #left_eye_hull = cv.convexHull(Lefteye)
            #right_eye_hull = cv.convexHull(Righteye)

            # 绘制轮廓
            #cv.drawContours(frame, [left_eye_hull], -1, [0, 255, 0], 1)
            #cv.drawContours(frame, [right_eye_hull], -1, [0, 255, 0], 1)

            # 眨眼判断
            if sysdic['mean_Ratio'] < sysdic['Radio']:
                sysdic['frame_counter'] += 1
            else:
                if sysdic['frame_counter'] >= sysdic['Low_radio_constant']:
                    sysdic['blink_counter'] += 1
                    sysdic['frame_counter'] = 0
                    sysdic['blink_counter_temp'] += 1
                if sysdic['blink_counter_temp'] == 1:
                    sysdic['time_start'] = time.time()
                if sysdic['blink_counter_temp'] >= sysdic['himachukaisu']:
                    sysdic['blink_counter_temp'] = 0
                    sysdic['time_start'] = time.time()
                    recdic['yobikaisu'] = 0

        recdic['EndTime'] = time.time()
        #print(str(callmanrunning)+" "+str(blink_counter_temp)+" "+str(time_end - time_start))
        if sysdic['blink_counter_temp'] > sysdic['chuuichukaisu']:
            if (recdic['EndTime'] - sysdic['time_start']) < sysdic['chuuibyou']:
                threading.Thread(target=self.callman,args=[recdic['yobikaisu']+1,recdic,sysdic]).start()
                recdic['yobikaisu'] += 1
                sysdic['blink_counter_temp'] = 0 



    def __calculate_Ratio(self,eye):
        """
        计算眼睛横纵比
        """
        d1 = distance.euclidean(eye[1], eye[5])
        d2 = distance.euclidean(eye[2], eye[4])
        d3 = distance.euclidean(eye[0], eye[3])
        d4 = (d1 + d2) / 2
        ratio = d4 / d3
        return ratio