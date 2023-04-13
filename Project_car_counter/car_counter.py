from ultralytics import YOLO
import cv2
import cvzone
import math
import torch
from sort import *
import numpy as np

cap = cv2.VideoCapture("../Videos/cars.mp4")
device = "cuda" if torch.cuda.is_available() else  "cpu"
print(device)

model = YOLO("../Yolo_weights/yolov8l.pt")
ClassNames = [model.names[name] for name in  model.names]
# ClassNames = {
# ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
#  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
#  'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
#  'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
#  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# }
print("Import Yolov8l")
mask = cv2.imread("mask.png")

#tracking
tracker = Sort(max_age = 20 , min_hits=3 ,iou_threshold=0.3)
# x1 , y1 , x2 , y2
limits = [400, 297, 673, 297]
totalCount = []
while True:
    success , img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)

    imgGraphics = cv2.imread("graphics.png",cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img,imgGraphics,(0,0))

    result = model(imgRegion, stream=True)
    detections = np.empty((0,5))
    #result 中有很多的Feature 
    for r in result :
        boxes = r.boxes
        for box in boxes :
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1) , int(y1) , int(x2) ,int(y2)
            # print(x1,y1,x2,y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h),l=9,t=2)
            conf = box.conf[0]
            print(f"confidence : {conf*100:.2f} %")
            #class name
            cls = box.cls[0]
            cls_name = ClassNames[int(cls)]
            conf = math.ceil(box.conf[0]*100) /100
            if cls_name == 'car' or cls_name == 'truck' or cls_name == 'bus'\
                    or cls_name == 'motorbike' and  conf > 0.3 :
                # cvzone.putTextRect(img,f"{cls_name} : {conf*100:.2f}",(max(0,x1),max(35,y1)),scale=1,thickness=2,offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, t=2,rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    #draw a line to count the number of cars
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
    for result in resultsTracker :
        x1,y1,x2,y2,Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w , h = x2-x1 , y2-y1
        cvzone.cornerRect(img , (x1,y1,w,h) , l=9 , rt=5 , colorR=(255,0,0))
        cvzone.putTextRect(img, f"[{int(Id)}]", (max(0, x1), max(35, y1)), scale=1, thickness=2,
                           offset=3)
        cx, cy = x1 + w//2 , y1 + h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        if limits[0]<cx<limits[2] and limits[1] - 30 < cy < limits[1]+30 :
            if totalCount.count(Id) == 0 :
                totalCount.append(Id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # cvzone.putTextRect(img,f"Counts = {len(totalCount)}" , (50,50))
    cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
    cv2.imshow("Image",img)
    cv2.imshow("ImageRegion", imgRegion)
    #0 for press to move on else auto move on
    cv2.waitKey(0)
