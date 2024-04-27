import cv2
import numpy as np
from bike_helmet_detector_image import detection2
from utils import visualization_utils as vis_util

# Load the video capture
cap = cv2.VideoCapture('test1.mp4')  # Change 'test1.mp4' to your video file
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Time interval for processing frames (in seconds)
processing_interval = 1

# Variables for timing
prev_time = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
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
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes_motorbike),
            np.squeeze(classes_motorbike).astype(np.int32),
            np.squeeze(scores_motorbike),
            category_index_motorbike,
            use_normalized_coordinates=True,
            line_thickness=2,
            min_score_thresh=0.70)

        k = vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes_helmet),
            np.squeeze(classes_helmet).astype(np.int32),
            np.squeeze(scores_helmet),
            category_index_helmet,
            use_normalized_coordinates=True,
            line_thickness=2,
            min_score_thresh=0.0)

        # Write the frame with detections to the output video
        out.write(k)
        cv2.imshow('Object detector', k)

        # Check if there are any "with_helmet" boxes with a score of 0%
        for box_helmet, score_helmet in zip(np.squeeze(boxes_helmet), np.squeeze(scores_helmet)):
            if category_index_helmet[int(classes_helmet[0][0])]['name'] == 'with_helmet' and score_helmet == 0:
                # Check if this "with_helmet" box is inside any "person" box
                for box_person in np.squeeze(boxes_motorbike):
                    if category_index_motorbike[int(classes_motorbike[0][0])]['name'] == 'person':
                        ymin_person, xmin_person, ymax_person, xmax_person = box_person
                        ymin_helmet, xmin_helmet, ymax_helmet, xmax_helmet = box_helmet
                        # Check if the helmet box is inside the person box
                        if ymin_person <= ymin_helmet <= ymax_person and xmin_person <= xmin_helmet <= xmax_person:
                            # Perform the desired action when a with_helmet box is inside a person box with 0% score
                            print("Alert: A person with a helmet detected with 0% score")
                            # You can add your alert mechanism here
                            break  # Break inner loop if a with_helmet box is found inside a person box


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
