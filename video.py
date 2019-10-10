#! /usr/bin/python
#coding:utf-8

import cv2
import numpy as np
import os.path as path
import argparse
import sys
import threading

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm", help = "m (or nothing) for meanShift and c for camshift")
args = vars(parser.parse_args())

def center(points):
    """计算给定矩阵的质心"""
    x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
    y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
    return np.array([np.float32(x), np.float32(y)], np.float32)

font = cv2.FONT_HERSHEY_SIMPLEX

class RTSCapture(cv2.VideoCapture):
    """Real Time Streaming Capture.
    """

    _cur_frame = None
    _reading = False

    @staticmethod
    def create(url):
        """这个类必须使用 RTSCapture.create 方法创建，请不要直接实例化"""
        rtscap = RTSCapture(url)
        #rtscap.frame_receiver = threading.Thread(target=rtscap.recv_frame, daemon=True)
        rtscap.frame_receiver = threading.Thread(target=rtscap.recv_frame)
        if isinstance(url, str) and url.startswith(("rtsp://", "rtmp://")):
            rtscap._reading = True

        return rtscap

    def isStarted(self):
        """替代 VideoCapture.isOpened() """
        ok = self.isOpened()
        if ok and self._reading:
            ok = self.frame_receiver.is_alive()
        return ok

    def recv_frame(self):
        """子线程读取最新视频帧方法"""
        while self._reading and self.isOpened():
            ok, frame = self.read()
            if not ok: break
            self._cur_frame = frame
        self._reading = False

    def read2(self):
        """读取最新视频帧
        返回结果格式与 VideoCapture.read() 一样
        """
        frame = self._cur_frame
        self._cur_frame = None
        return frame is not None, frame

    def start_read(self):
        """启动子线程读取视频帧"""
        self.frame_receiver.start()
        self.read_latest_frame = self.read2 if self._reading else self.read

    def stop_read(self):
        """退出子线程方法"""
        self._reading = False
        if self.frame_receiver.is_alive(): self.frame_receiver.join()

class Pedestrian():

  def __init__(self, id, frame, track_window):
    # 建立ROI
    self.id = int(id)
    x,y,w,h = track_window
    self.track_window = track_window
    self.roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([self.roi], [0], None, [16], [0, 180])
    self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # s建立kalman滤波器
    self.kalman = cv2.KalmanFilter(4,2)
    self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
    self.measurement = np.array((2,1), np.float32) 
    self.prediction = np.zeros((2,1), np.float32)
    self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    self.center = None
    self.update(frame)
    
  def __del__(self):
    print("Pedestrian %d destroyed" % self.id)

  def update(self, frame):
    print("updating %d " % self.id)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    back_project = cv2.calcBackProject([hsv],[0], self.roi_hist,[0,180],1)
    
    if args.get("algorithm") == "c":
      ret, self.track_window = cv2.CamShift(back_project, self.track_window, self.term_crit)
      pts = cv2.boxPoints(ret)
      pts = np.int0(pts)
      self.center = center(pts)
      cv2.polylines(frame,[pts],True, 255,1)
      
    if not args.get("algorithm") or args.get("algorithm") == "m":
      ret, self.track_window = cv2.meanShift(back_project, self.track_window, self.term_crit)
      x,y,w,h = self.track_window
      self.center = center([[x,y],[x+w, y],[x,y+h],[x+w, y+h]])  
      cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 2)

    self.kalman.correct(self.center)
    prediction = self.kalman.predict()
    cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 4, (255, 0, 0), -1)
    # 假影
    cv2.putText(frame, "ID: %d -> %s" % (self.id, self.center), (11, (self.id + 1) * 25 + 1), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    # 实际信息
    cv2.putText(frame, "ID: %d -> %s" % (self.id, self.center), (10, (self.id + 1) * 25), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

def main():
  
  camera = RTSCapture.create(sys.argv[1])
  camera = RTSCapture.create('rtsp://admin:yhj681970@192.168.1.64:554/h265/ch1/sub/av_stream')
  camera.start_read()
  
  history = 20
  
  # KNN背景减法器
  bs = cv2.createBackgroundSubtractorKNN()

  # MOG背景减法器
  # bs = cv2.bgsegm.createBackgroundSubtractorMOG(history = history)
  # bs.setHistory(history)

  # GMG背景减法器
  # bs = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames = history)
  
  cv2.namedWindow("surveillance")
  pedestrians = {}
  firstFrame = True
  frames = 0
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter('output.avi',fourcc, 15.0, (640,480))
  while camera.isStarted():
    print(" -------------------- FRAME %d --------------------" % frames)
    grabbed, frane = camera.read()
    if (grabbed is False):
      print("failed to grab frame.")
      break

    ret, frame = camera.read()
    fgmask = bs.apply(frame)

    if frames < history:
      frames += 1
      continue

    th = cv2.threshold(fgmask.copy(), 127, 255, cv2.THRESH_BINARY)[1]
    th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 2)
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,3)), iterations = 2)
    # image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    counter = 0
    for c in contours:
      if cv2.contourArea(c) > 500:
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 1)
        if firstFrame is True:
          pedestrians[counter] = Pedestrian(counter, frame, (x,y,w,h))
        counter += 1

    for i, p in pedestrians.iteritems():
      p.update(frame)
    
    firstFrame = False
    frames += 1

    cv2.imshow("surveillance", frame)
    out.write(frame)
    if cv2.waitKey(110) & 0xff == 27:
        break
        
  camera.stop_read()
  out.release()
  camera.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
