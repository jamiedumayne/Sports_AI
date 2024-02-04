import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

os.chdir("C:/Users/jamie/Files/Code/sports_tracker/tennis_tracker/resources")

model = YOLO("yolov8n-seg.pt")
names = model.model.names

frame_number = 1
frame_time = 1
frame_number_arr, top_arr, bot_arr, left_arr, right_arr, car_num = [], [], [], [], [], []
frame2_number_arr, top2_arr, bot2_arr, left2_arr, right2_arr, car2_num = [], [], [], [], [], []
video_title = cv2.VideoCapture("tennis_black_boxed.mp4")

out = cv2.VideoWriter('tennis_vid_box.avi',
                      cv2.VideoWriter_fourcc(*'MJPG'),
                      30, (int(video_title.get(3)), int(video_title.get(4))))

while True:
    ret, frame = video_title.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.predict(frame)

    #runs even if there is no detection
    if results[0].masks != None:
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy

        x_values = []
        y_values = []
        top_val, bot_val, right_val, left_val = 0, 0, 0, 0

        points = masks[0]

        for each_point in points:
            x_values.append(each_point[0])
            y_values.append(each_point[1])

        bot_val = int(max(y_values))
        top_val = int(min(y_values))
        right_val = int(max(x_values))
        left_val = int(min(x_values))

        annotator = Annotator(frame, line_width=2)

        for mask, cls in zip(masks, clss):
            annotator.seg_bbox(mask=mask,
                               mask_color=colors(int(cls), True),
                               det_label=names[int(cls)])

        #draw a rectangle around car
        for test in masks:
            cv2.rectangle(frame, (left_val,top_val), (right_val, bot_val), (0,255,0), 5)

        """
        if len(masks) > 1:
            points = masks[1]

            x2_values = []
            y2_values = []
            top2_val, bot2_val, right2_val, left2_val = 0, 0, 0, 0

            for each_point in points:
                x2_values.append(each_point[0])
                y2_values.append(each_point[1])

            bot2_val = int(max(y2_values))
            top2_val = int(min(y2_values))
            right2_val = int(max(x2_values))
            left2_val = int(min(x2_values))

            annotator = Annotator(frame, line_width=2)

            for mask, cls in zip(masks, clss):
                annotator.seg_bbox(mask=mask,
                                   mask_color=colors(int(cls), True),
                                   det_label=names[int(cls)])

            #draw a rectangle around car
            for test in masks:
                cv2.rectangle(frame, (left2_val,top2_val), (right2_val, bot2_val), (0,255,0), 5)

        else:
            bot2_val = 0
            top2_val = 0
            right2_val = 0
            left2_val = 0
        """
        bot2_val = 0
        top2_val = 0
        right2_val = 0
        left2_val = 0
            

    car_num.append("car1")
    frame_number_arr.append(frame_number)
    top_arr.append(top_val)
    bot_arr.append(bot_val)
    left_arr.append(left_val)
    right_arr.append(right_val)

    car2_num.append("car2")
    frame2_number_arr.append(frame_number)
    top2_arr.append(top2_val)
    bot2_arr.append(bot2_val)
    left2_arr.append(left2_val)
    right2_arr.append(right2_val)


    frame_number = frame_number + 1

    out.write(frame)
    cv2.imshow("instance-segmentation", frame)

    if cv2.waitKey(frame_time) & 0xFF == ord('q'):
        break

out.release()
video_title.release()
cv2.destroyAllWindows()

#combine arrays to df and save as csv
dataset1 = pd.DataFrame({'Car': car_num, 'Frame': frame_number_arr, 'top': top_arr, 'bot': bot_arr,
    'left': left_arr, 'right': right_arr})

dataset2 = pd.DataFrame({'Car': car2_num, 'Frame': frame2_number_arr,
 'top': top2_arr, 'bot': bot2_arr, 'left': left2_arr, 'right': right2_arr})

dataset = pd.concat([dataset1, dataset2])

dataset.to_csv('tennis_test.csv', index=False)


print("Finished")


#to do
#figure out how to handle multiple players