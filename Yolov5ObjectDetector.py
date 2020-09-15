#
# Yolov5ObjectDetector.py
# 
# This is based on yolov5/detect.py implemenation
# on https://github.com/ultralytics/yolov5
#
#  2020/09/01 antillia.com (C) Toshiyuki Arai


# encoding: utf-8

import argparse
import os
import sys
import traceback

import platform

import shutil
import time
from pathlib import Path

#2020/08/31 to avoid import error of tlinker
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

from FiltersParser import FiltersParser 


class Yolov5ObjectDetector:
  # Constructor
  def __init__(self, imgsz=1024, device='', weights='yolov5m.pt'):
      self.imgsz   = imgsz
      self.weights = weights
      self.device  =  device # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
      set_logging()
      self.device = select_device(self.device)

      self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

      # Load model
      self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
      imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size
      print("image size {}".format(imgsz))
      self.augment = False
      
      if self.half:
          self.model.half()  # to FP16

      # Get names and colors
      self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
      #print("Class names {}".format(names))
      
      self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

      #Parameters fo NMS      
      self.conf_thres   = 0.4
      self.iou_thres    = 0.5
      self.classes      = None
      self.agnostic_nms = False


  def detect(self, filters, source, output):
    #
    with torch.no_grad():

      print("detect filters {}".format(filters))
      
      self.output = output
      
      self.webcam = source.isnumeric() or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

      #print("opt.img_size{}".format(opt.img_size))
      


      # Set Dataloader
      self.dataset = None
      
      vid_path, vid_writer = None, None
      if self.webcam:
          view_img = True
          cudnn.benchmark = True  # set True to speed up constant image size inference
          self.dataset = LoadStreams(source, img_size=imgsz)
      else:
          save_img = True
          self.dataset = LoadImages(source, img_size=self.imgsz)


      # Run inference
      t0 = time.time()
      img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
      _ = self.model(img.half() if half else img) if self.device.type != 'cpu' else None  # run once
      for path, img, im0s, vid_cap in self.dataset:
          img = torch.from_numpy(img).to(self.device)
          img = img.half() if self.half else img.float()  # uint8 to fp16/32
          img /= 255.0  # 0 - 255 to 0.0 - 1.0
          if img.ndimension() == 3:
              img = img.unsqueeze(0)

          # Inference
          self.t1 = time_synchronized()
          pred = self.model(img, augment=self.augment)[0]

          # Apply NMS
          #pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
          pred = self.apply_nms(pred)
          
          # Visualize detections
          #saved_files = (saved_image_file, saved_objects_csvfile, saved_stats_csvfile) 
          saved_triple_files = self.visualize(filters, path, img, im0s, pred)
          
          print('Results saved to %s' % Path(self.output))
          if platform.system() == 'Darwin' and not opt.update:  # MacOS
             os.system('open ' + save_path)

          print('Done. (%.3fs)' % (time.time() - t0))
          print(saved_triple_files)
          
          return saved_triple_files 


  def apply_nms(self, pred):
      pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms)
      return pred
      
          
  def visualize(self, filters, path, img, im0s, pred):
       save_txt = True
       save_img = True
       view_img = False
       

       t2 = time_synchronized()
       print("visualize ")
       
       for i, det in enumerate(pred):  # detections per image
           if self.webcam:  # batch_size >= 1
               p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
           else:
               p, s, im0 = path, '', im0s

           save_path = str(Path(self.output) / Path(p).name)
           txt_path  = str(Path(self.output) / Path(p).stem) +  ('_%g' % self.dataset.frame if self.dataset.mode == 'video' else '')
           #s += '%gx%g ' % img.shape[2:]  # print string
           
           s  = ""
           imsize= img.shape[2:]
           print("Image_shape {}".format(imsize))
           
           gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
           if det is not None and len(det):
               # Rescale boxes from img_size to im0 size
               det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

               # Print results
               for c in det[:, -1].unique():
                   n = (det[:, -1] == c).sum()  # detections per class
                   class_name = self.names[int(c)]
                   #2020/09/01
                   #Apply the filters
                   if filters != None:
                      if class_name in filters:
                        s += '%ss, %g \n' % (self.names[int(c)], n)  # add to string
                   else:
                     s += '%ss, %g \n' % (self.names[int(c)], n)  # add to string

               # Write results
               
               i = 1
               detected_objects = []

               for *xyxy, conf, cls in reversed(det):
                   if save_txt:  # Write to file
                       xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                       #with open(txt_path + '.csv', 'w') as f:
                       name = self.names[int(cls)]
                       cf = '%.2f'%(conf)
                       c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                       #print("{} {}".format(c1, c2))
                       x, y   = c1
                       x2, y2 = c2
                       w = x2 - x
                       h = y2 - y

                   if save_img or view_img:  # Add bbox to image
                       class_name = self.names[int(cls)]
                       #label = '%s %.2f' % (self.names[int(cls)], conf)
                       #2020/09/01 Apply filters
                       if filters != None:
                           if class_name in filters:
                              label = '%s %s' % (i, class_name)

                              data = "{}, {}, {}, {}, {}, {}, {}\n".format(str(i), name, str(cf), x, y, w, h)
                              
                              detected_objects.append(data)
                              #print("class_name '{}' filters {}".format(class_name, filters))
                              plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)
                              i += 1

                           else:
                              pass
                       else:
                          label = '%s %s' % (i, class_name)

                          data = "{}, {}, {}, {}, {}, {}, {}\n".format(str(i), name, str(cf), x, y, w, h)
                          detected_objects.append(data)

                          plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)
                          i += 1

           filters_name = ""
                
           if filters is not None:
             parser = FiltersParser(str(filters))
             filters_name = parser.get_filters_name()
              
           
           #print("text_path {}".format(txt_path))
           arr = os.path.split(txt_path)
           #print(arr)
           
           #filtered_filename    = filters_name + arr[1]
           filtered_output_path = txt_path + filters_name # os.path.join(arr[0], filtered_filename)
           
           
           print("Output file path {}".format(filtered_output_path))
           
           saved_objects_csvfile = self.save_detected_objects_as_csvfile(filtered_output_path, detected_objects)

           saved_stats_csvfile   = self.save_stats_as_csvfile(filtered_output_path, s)
                                        
           # Print time (inference + NMS)
           print('%sDone. (%.3fs)' % (s, t2 - self.t1))


           # Save results (image with detections)
           dir, basename = os.path.split(save_path)
           
           basename_without_ext, ext = os.path.splitext(basename)
           filtered_image_filename = basename_without_ext + filters_name + ext
           
           saved_image_file = os.path.join(dir, filtered_image_filename)
                      
           if save_img:
               if self.dataset.mode == 'images':
                   print("Saved image to {}".format(saved_image_file))
                   
                   cv2.imwrite(saved_image_file, im0)
               else:
                   if vid_path != saved_image_file:  # new video
                       vid_path = saved_image_file
                       if isinstance(vid_writer, cv2.VideoWriter):
                           vid_writer.release()  # release previous video writer

                       fourcc = 'mp4v'  # output video codec
                       fps = vid_cap.get(cv2.CAP_PROP_FPS)
                       w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                       h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                       vid_writer = cv2.VideoWriter(saved_image_file, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                   vid_writer.write(im0)
       #save_txt = True

       saved_triple_files = (saved_image_file, saved_objects_csvfile, saved_stats_csvfile)
       return saved_triple_files
      
      
 
  def save_detected_objects_as_csvfile(self, txt_path, detected_objects):
       #2020/09/01                      
       #Save detected_objects data to a csv file.
       header = "id, class, score, x, y, w, h\n"
       objects_csv_filepath = txt_path + '.csv'
       with open(objects_csv_filepath, 'w') as f:
           f.write(header)
           for item in detected_objects:
             #print(item)
             f.write(item)
       print("Saved a detected_objects_csv_file {}".format(objects_csv_filepath))
       return objects_csv_filepath
       
       
  def save_stats_as_csvfile(self, txt_path, stats):
      #Save a stats csv file.
      header = "class, count\n"
      stats_csv_filepath = txt_path + '_stats' + '.csv'
      with open(stats_csv_filepath, 'w') as f:
         f.write(header)
         f.write(stats)
      print("Saved a stats_csv_file {}".format(stats_csv_filepath))
      return stats_csv_filepath




if __name__ == '__main__':

  try:
     if len(sys.argv) < 3:
        raise Exception("Usage: {} source output_dir filters".format(sys.argv[0]))
        
     source  = None
     output  = None
     filters = None  # classnames_list something like this "[person,car]"
     
     if len(sys.argv) >= 2:
       source = sys.argv[1]
 
     if len(sys.argv) >= 3:
       output = sys.argv[2]
       if not os.path.exists(output):
          os.makedirs(output)
         
     if len(sys.argv) == 4:
       str_filters = sys.argv[3]
       
       filtersParser = FiltersParser(str_filters)
       filters = filtersParser.get_filters()
          
     print("source {}".format(source))
     print("output {}".format(output))
     print("filters {}".format(filters))
 
     saved_triple_files = None
     with torch.no_grad():
       detector = Yolov5ObjectDetector()
       saved_triple_files = detector.detect(filters, source, output)

  except:
    traceback.print_exc()
