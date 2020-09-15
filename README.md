<html>
<body>
<h1>Yolov5ObjectDetector</h1>
<font size=3><b>
This is a simple python class Yolov5ObjectDetector based on yolov5/detect.py implementation
on https://github.com/ultralytics/yolov5.<br>
</b></font>
<br>
<h2>1 Installation </h2>
<h3>
1.1 Yolov5ObjectDetector
</h3>
<font size=2>
 We have downloaded <a href="https://github.com/ultralytics/yolov5">yolov5</a>.
and installed pytorch-cpu and tochvision-cpu in the following way.<br>

<br>
<table style="border: 1px solid red;">
<tr><td>
<font size=2>
git clone https://github.com/ultralytics/yolov5.git<br>
cd yolov5<br>

pip install -r requirements.txt
</font>
</td></tr>
</table>
<br>

Please download Yolov5ObjectDetector.git from https://github.com/atlan-antillia/Yolov5ObjectDetector repository to your working folder.<br><br>


somewhere>git clone https://github.com/atlan-antillia/Yolov5ObjectDetector.git<br>
cd Yolov5ObjectDetector<br>
<br>
Yo may see the following files in the Yolov5ObjectDetector:<br>
<br>
FiltersParser.py<br>
Yolov5ObjectDetector.py<br>
images/<br>
output/<br>
yolov5m.pt<br>

Please copy these files to yolov5 folder.<br><br>


<h3>
1.2 How to run Yolov5ObjectDetector
</h3>

Please run the following command.<br>

yolov5>python Yolov5ObjectDetector.py input_image_file_or_dir  output_image_dir [optional_filters]
<br>
<br>
<b>
Example 1:<br>
yolov5>python Yolov5ObjectDetector.py images\img.png output <br>
</b>
 The above command will generate a triplet of files (detected_objects_image, detected objects_detail, detected_objects_stats), 
 and save them as the output diretory.<br>

output/img.png<br>
<img src = "./output/img.png" width="1024" height="auto">
<br>
<br>

ouput/img.csv<br>
<img src = "./output/img.csv.png" >
<br>
<br>
output/img_stats.csv<br>
<img src = "./output/img_stats.csv.png" >
<br>
<br>
<b>
Example 2: filers=[person,car]<br>
</b>
yolov5>python Yolov5ObjectDetector.py images\img.png output [person,car]<br>
output/img_person_car.png<br>
<img src = "./output/img_person_car.png" width="1024" height="auto">
<br>
<br>

<br>
<br>

ouput/img_person_car.csv<br>
<img src = "./output/img_person_car.csv.png" >
<br>
<br>
output/img_person_car_stats.csv<br>
<img src = "./output/img_person_car_stats.csv.png" >
<br>
<br>

</body>
</html>

