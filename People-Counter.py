import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("Videos/Traffic2.mp4")  # For Video

model = YOLO("../Yolo-Weights/yolov8m.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

#mask = cv2.imread("maSk.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limitsSin = [380, 180, 596, 180]
limitsSout = [10, 360, 550, 360]
limitsNout = [650, 180, 990, 180]
limitsNin = [750, 360, 2010, 360]
limitsO = [10, 100, 50, 100]

totalCountSin = []
totalCountSout = []
totalCountNin = []
totalCountNout = []
toto=[]

while True:
    success, img = cap.read()
    # imgRegion = cv2.bitwise_and(img, mask)

    # imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    # img = cvzone.overlayPNG(img, imgGraphics, (730, 260))
    results = model(img, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike"\
                    and conf > 0.2:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limitsSin[0], limitsSin[1]), (limitsSin[2], limitsSin[3]), (255,0, 0),2)
    cv2.line(img, (limitsSout[0], limitsSout[1]), (limitsSout[2], limitsSout[3]), (0, 0, 255), 2)
    cv2.line(img, (limitsNin[0], limitsNin[1]), (limitsNin[2], limitsNin[3]), (255,0, 0),2)
    cv2.line(img, (limitsNout[0], limitsNout[1]), (limitsNout[2], limitsNout[3]), (0, 0, 255), 2)
    cv2.line(img, (limitsO[0], limitsO[1]), (limitsO[2], limitsO[3]), (0, 0, 255), 2)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                           scale=1, thickness=2, offset=3)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limitsSin[0] < cx < limitsSin[2] and limitsSin[1] - 15 < cy < limitsSin[1] + 15:
            if totalCountSin.count(id) == 0:
                totalCountSin.append(id)
                cv2.line(img, (limitsSin[0], limitsSin[1]), (limitsSin[2], limitsSin[3]), (0, 255, 0), 5)

        if limitsSout[0] < cx < limitsSout[2] and limitsSout[1] - 20 < cy < limitsSout[1] + 20:
            if totalCountSout.count(id) == 0:
                totalCountSout.append(id)
                cv2.line(img, (limitsSout[0], limitsSout[1]), (limitsSout[2], limitsSout[3]), (0, 255, 0), 5)

        if limitsNin[0] < cx < limitsNin[2] and limitsNin[1] - 15 < cy < limitsNin[1] + 15:
            if totalCountNin.count(id) == 0:
                totalCountNin.append(id)
                cv2.line(img, (limitsNin[0], limitsNin[1]), (limitsNin[2], limitsNin[3]), (0, 255, 0), 5)

        if limitsNout[0] < cx < limitsNout[2] and limitsNout[1] - 20 < cy < limitsNout[1] + 20:
            if totalCountNout.count(id) == 0:
                totalCountNout.append(id)
                cv2.line(img, (limitsNout[0], limitsNout[1]), (limitsNout[2], limitsNout[3]), (0, 255, 0), 5)

        if limitsO[0] < cx < limitsO[2] and limitsO[1]-10 < cy < limitsO[1]+10:
            if toto.count(id) == 0:
                toto.append(id)
                cv2.line(img, (limitsO[0], limitsO[1]), (limitsO[2], limitsO[3]), (0, 255, 0), 5)
    # # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img,f'Total in (S): {str(len(totalCountSin))}',(929,80),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
    cv2.putText(img, f'Total out (S): {str(len(totalCountSout))}', (929, 120), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    cv2.putText(img,f'Inframe(S):{str(len(totalCountSin)-len(totalCountSout))}',(929,150),cv2.FONT_HERSHEY_PLAIN,2,(50,50,230),2)

    cv2.putText(img, f'Total in (N): {str(len(totalCountNin))}', (929, 205), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0),2)
    cv2.putText(img, f'Total out (N): {str(len(totalCountNout))}', (929, 235), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    cv2.putText(img, f'Total (N+S):{str((len(totalCountNout) + len(totalCountSout)))}', (929, 340), cv2.FONT_HERSHEY_PLAIN,2, (10, 150, 30), 2)

    cv2.putText(img, f'Total in (Side): {str(len(toto))}', (20, 90), cv2.FONT_HERSHEY_PLAIN,2, (0,0,255),2)
    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(0)
