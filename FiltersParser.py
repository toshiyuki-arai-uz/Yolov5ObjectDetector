# FiltersParser.py
import os
import sys

class FiltersParser:

  # Specify a str_filters string like this "[person,motorcycle]" ,
  def __init__(self, str_filters): #No check classes , classes=COCO_CLASSES):
      print("FiltersParser {}".format(str_filters))
            
      self.str_filters  = str_filters
      #self.classes      = classes
      self.filters  = []


  def get_filters(self):
      self.filters = []
      if self.str_filters != None:
          tmp = self.str_filters.strip('[]').split(',')
          if len(tmp) > 0:
              for e in tmp:
                  e = e.lstrip()
                  e = e.rstrip()
                  #No check classes 
                  #if e in self.classes :
                  self.filters.append(e)
      if self.filters is not None and len(self.filters) == 0:
          self.filters = None

      return self.filters

  def get_filters_name(self):
        filname = ""
        if self.str_filters is not None:
            filname = "_"
            filname += self.str_filters.strip("[]").replace("'", "").replace(", ", "_")
            #if len(filname) != 0:
            #  filname += "_"
        return filname

  #2020/07/31 Updated 
  def get_ouput_filename(self, input_image_filename, image_out_dir):
        rpos  = input_image_filename.rfind("/")
        fname = input_image_filename
        if rpos >0:
            fname = input_image_filename[rpos+1:]
        else:
            rpos = input_image_filename.rfind("\\")
            if rpos >0:
                fname = input_image_filename[rpos+1:]
          
        
        abs_out  = os.path.abspath(image_out_dir)
        if not os.path.exists(abs_out):
            os.makedirs(abs_out)

        filname = ""
        if self.str_filters is not None:
            filname = self.str_filters.strip("[]").replace("'", "").replace(", ", "_")
            if len(filname) != 0:
              filname += "_"
        
        output_image_filename = os.path.join(abs_out, filname + fname)
        return output_image_filename
