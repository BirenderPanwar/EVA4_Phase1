# Assignment 13-2: Training custom dataset for YOLOV3

New Class used: Flying Disc
--------------

Refered resources:
------------------

- colab: https://colab.research.google.com/drive/1LbKkQf4hbIuiUHunLlvY-cc0d_sNcAgS
- GitHub: https://github.com/theschoolofai/YoloV3
- https://www.y2mate.com/en19 (for downloading youtube video)
- https://github.com/miki998/YoloV3_Annotation_Tool [annotations tool]
- https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence [FFMPEG tool]

Steps Followed:
---------------

1. 500 unique dataset collection for class not present in YoloV3
2. Annotations
3. Configuration for yolov3-custom.cfg
4. custom data preperation
5. training
6. video frame genertaion for youtube video using ffmpeg tool
7. applying detect.py on video frame for object detection. it generate frames with predicted bounding boxes
8. generate final yolo video for frame having predicted bounding boxes
9. upload video on youtube.


Configuration for yolov3-custom.cfg
-----------------------------------

path: /cfg/yolov3-custom.cfg

Following Two changes are done in .cfg file. classes=1 and model only one class dataset:

- Filter size for last convolution layer is set as 18 [(classes(1) + objectiveness(1) + bbox(4)) * num of anchor boxes(3)]
- classes=1 

Custom dataset Preperation and setup
------------------------------------

- 500 unique images are downloaded from internet
- Each images filename are set in orger for smooth operation as img001.jpg, img002.jpg to img500.jpg
- all images are placed under /data/customdata/images
- All images are annotated using YoloV3_Annotation_Tool. For each images correcponding annotation file is saved as .txt with same file name.
- All annotations files/labels are placed under /data/customdata/labels
- images file name and its lables file name shall be same. Example: img001.jpg<==>img001.txt

Sample dataset collected from internet
--------------------------------------

![](samples.PNG)


Model Weight Folder:
--------------------

- weight folder is created under Yolov3 root
- yolov3.weights and yolov3-spp-ultralytics.pt are place under this "weight" folder


Model Training:
---------------

invoke train.py with proper -data and --cfg argument as below:

![](training_cmd.PNG)


# Bounding box and class prediction for video frame downloded from youtube
---------------------------------------------------------------------------

1. Short video having flying dish object class is downloaded from youtube using y2mate tool

2. ffmpeg tool is used to get the frames from mp4 video. all video frame is stored under "video_frames" folder

- Following command is used to pick frames for specific time interval:

ffmpeg -i inp_video.mp4 -ss 00:02:20 -to 00:02:50 image-%04d.jpg

3. detect.py is applied on input video frame and output frames are created with predicted bounding boxes

4. ffmpeg tool is then used to generate the output video(.mp4) from output frames

ffmpeg -i image-%04d.jpg flying_disc_yolo.mp4


Result: 
-------

Final Yolov3 object detection video for detecing "flying disc" is uploaded on YouTube.

Link: 

----------------------------------------------------------------------------------------------------------------

