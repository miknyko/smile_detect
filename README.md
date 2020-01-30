# 笑容检测
Detecting the simle face in a video stream (your own camera)  
使用opencv中预训练好的res10网络侦测视频流中的人脸，再使用笑容检测器侦测笑容。  
笑脸检测器为一个Minivgg，训练数据来源于网络自整理，效果不错。
视频结束有一个得分，使用简单算法告诉你整个过程的快乐值


## 运行环境:
  python = 3.6  
  tensorflow =1.14  
  imutils  
  opencv-contrib-python
  
## 使用方法：
安装好所需要环境后，运行run.py，注意需要具备摄像头。  
按q退出脚本


## 应用场景：

打开一部喜剧片，打开这个程序，看完电影，看看你是否得到了欢乐？

Find a comedy movie,and turn on the camera,see how much happiness you gained!
