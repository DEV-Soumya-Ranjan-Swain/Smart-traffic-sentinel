import cv2
import numpy as np
from bike_helmet_detector_image import detection2
from utils import visualization_utils as vis_util

import requests
# for i in range(1,31):
def detect_number(frame):
    regions = ['in']
    str1 =frame

    with open(str1, 'rb') as fp:
        response = requests.post('https://api.platerecognizer.com/v1/plate-reader/',
                                 data=dict(regions=regions), files=dict(upload=fp),
                                 headers={'Authorization': 'Token 632bb3580c827e9bc7bec59319105ca21d79a14e'})

    results = response.json().get('results', [])  # Get the list of results or an empty list if not found

    if results:  # Check if there are any results
        plate_number = results[0]['plate']  # Access the plate number from the first result
        print(plate_number.upper())
    else:
        print("No plate numbers found.")


# Load the video capture
cap = cv2.VideoCapture('test4.mp4')  # Change 'test1.mp4' to your video file
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
# Time interval for processing frames (in seconds)
processing_interval = 1

# Variables for timing
prev_time = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    imh,imw,imc=frame.shape
    if not ret:
        break

    # Get the current time
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Convert milliseconds to seconds

    # Check if it's time to process this frame
    if current_time - prev_time >= processing_interval:
        # Update previous time
        prev_time = current_time

        # Detection for helmet
        category_index_helmet, _, boxes_helmet, scores_helmet, classes_helmet, _ = \
            detection2('frozen_graphs', '/frozen_inference_graph_helmet.pb', '/labelmap_helmet.pbtxt', 2, frame)

        # Detection for motorbike and person
        category_index_motorbike, _, boxes_motorbike, scores_motorbike, classes_motorbike, _ = \
            detection2('frozen_graphs', '/frozen_inference_graph_motorbike.pb', '/labelmap_motorbike.pbtxt', 4, frame)

        # Iterate over detected objects
        for box, score, classes, category_index in zip(
                [boxes_helmet, boxes_motorbike],
                [scores_helmet, scores_motorbike],
                [classes_helmet, classes_motorbike],
                [category_index_helmet, category_index_motorbike]
        ):

            for box_item, score_item, class_item in zip(box, score, classes):
                if category_index[int(class_item[0])]['name'] == 'person' and score_item >= 0.7:
                    ymin, xmin, ymax, xmax = box_item
                    print("Box Person")
                    print(box_item)
                    # Check if there are any "with_helmet" boxes inside this person box
                    without_helmet_detected = True
                    for box_helmet, score_helmet in zip(np.squeeze(boxes_helmet), np.squeeze(scores_helmet)):
                        if not np.all(box_helmet == 0) and category_index_helmet[int(classes_helmet[0][0])][
                            'name'] == 'with_helmet' and score_helmet >= 0.7:
                            ymin_helmet, xmin_helmet, ymax_helmet, xmax_helmet = box_helmet
                            print("Box Helmet")
                            print(box_helmet)
                            print(ymin_helmet, xmin_helmet, ymax_helmet, xmax_helmet)
                            # Check if the helmet box is inside the person box
                            if ymin < ymin_helmet < ymax or xmin < xmin_helmet < xmax or ymin < ymax_helmet < ymax or xmin < xmax_helmet < xmax:
                                without_helmet_detected = False
                                print("Helmet detected for this person.")

                    # Draw a red bounding box if no helmet is detected inside the person box
                    if without_helmet_detected:
                        print("No helmet detected for this person. Drawing red bounding box.")
                        start_point = (int(xmin * imw), int(ymin * imh))
                        end_point = (int(xmax * imw), int(ymax * imh))
                        color = (0, 0, 255)  # Red color
                        thickness = 2
                        image = cv2.rectangle(frame, start_point, end_point, color, thickness)
                        cv2.imshow('Object detector', image)
                        cropped = image[int(ymin * imh):int(ymax * imh), int(xmin * imw):int(xmax * imw)]
                        cv2.imwrite('cropped_image.jpg', cropped)
                        detect_number('cropped_image.jpg')

                elif category_index[int(class_item[0])]['name'] == 'motorbike' and score_item >= 0.4:
                    ymin, xmin, ymax, xmax = box_item
                    print("Box Motorbike")
                    print(box_item)
                    # Check if there are any "with_helmet" boxes inside this motorbike box
                    without_helmet_detected = True
                    for box_helmet, score_helmet in zip(np.squeeze(boxes_helmet), np.squeeze(scores_helmet)):
                        if not np.all(box_helmet == 0) and category_index_helmet[int(classes_helmet[0][0])][
                            'name'] == 'with_helmet' and score_helmet >= 0.7:
                            ymin_helmet, xmin_helmet, ymax_helmet, xmax_helmet = box_helmet
                            print("Box Helmet")
                            print(box_helmet)
                            print(ymin_helmet, xmin_helmet, ymax_helmet, xmax_helmet)
                            # Check if the helmet box is inside the motorbike box
                            if ymin < ymin_helmet < ymax or xmin < xmin_helmet < xmax or ymin < ymax_helmet < ymax or xmin < xmax_helmet < xmax:
                                without_helmet_detected = False
                                print("Helmet detected for this motorbike.")

                    # Draw a red bounding box if no helmet is detected inside the motorbike box
                    if without_helmet_detected:
                        print("No helmet detected for this motorbike. Drawing red bounding box.")
                        start_point = (int(xmin * imw), int(ymin * imh))
                        end_point = (int(xmax * imw), int(ymax * imh))
                        color = (0, 0, 255)  # Red color
                        thickness = 2
                        image = cv2.rectangle(frame, start_point, end_point, color, thickness)
                        cv2.imshow('Object detector', image)
                        cropped = image[int(ymin * imh):int(ymax * imh), int(xmin * imw):int(xmax * imw)]
                        cv2.imwrite('cropped_image.jpg', cropped)
                        detect_number('cropped_image.jpg')

            # Display the processed frame
            # cv2.imshow('Object detector', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Release resources
cap.release()
cv2.destroyAllWindows()
