import cv2
import numpy as np
from bike_helmet_detector_image import detection2
from utils import visualization_utils as vis_util
import requests

def give_me_out_res(video_given):
    
    def detect_number(frame):
        myBikeNumber= None
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
            myBikeNumber=plate_number.upper()
            return 1,myBikeNumber
        else:
            print("No plate numbers found.")
            return 0,myBikeNumber
    # Load the video capture
    cap = cv2.VideoCapture(video_given)  # Change 'test1.mp4' to your video file
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
        if(frame is None):
            return 0
        imh,imw,imc=frame.shape
        if not ret:
            break

        # Get the current time
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Convert milliseconds to seconds

        # Check if it's time to process this frame
        if current_time - prev_time >= processing_interval:
            # Update previous time
            

            # Detection for helmet
            category_index_helmet, _, boxes_helmet, scores_helmet, classes_helmet, _ = \
                detection2('frozen_graphs', '/frozen_inference_graph_helmet.pb', '/labelmap_helmet.pbtxt', 2, frame)

            # Detection for motorbike and person
            category_index_motorbike, _, boxes_motorbike, scores_motorbike, classes_motorbike, _ = \
                detection2('frozen_graphs', '/frozen_inference_graph_motorbike.pb', '/labelmap_motorbike.pbtxt', 4, frame)
            prev_time = current_time

            # Draw bounding boxes on the frame
            print("motorbike_box")
            print(boxes_motorbike)
            for box_person, score_person in zip(np.squeeze(boxes_motorbike), np.squeeze(scores_motorbike)):
                if classes_motorbike[0][0] in category_index_motorbike and category_index_motorbike[int(classes_motorbike[0][0])]['name'] == 'person' and not np.all(box_person == 0) and score_person >= 0.9:
                    ymin_person, xmin_person, ymax_person, xmax_person = box_person
                    print("box person")
                    print(box_person)
                    # Check if there are any "with_helmet" boxes inside this person box
                    without_helmet_detected = False
                    for box_helmet, score_helmet in zip(np.squeeze(boxes_helmet), np.squeeze(scores_helmet)):
                        # if category_index_helmet[int(classes_helmet[0][0])]['name'] == 'with_helmet' and score_helmet == 0 and score_person>=90 and scores_motorbike>=90:
                        if not np.all(box_helmet == 0) and category_index_helmet[int(classes_helmet[0][0])]['name'] == 'with_helmet' and score_helmet <= 0.1 and score_person >= 0.5 :
                            ymin_helmet, xmin_helmet, ymax_helmet, xmax_helmet = box_helmet
                            print("Box Helmet")
                            print(box_helmet)
                            print(ymin_helmet, xmin_helmet, ymax_helmet, xmax_helmet)
                            # Check if the helmet box is inside the person box
                            if ymin_person < ymin_helmet < ymax_person or xmin_person < xmin_helmet < xmax_person or ymin_person < ymax_helmet < ymax_person or xmin_person < xmax_helmet < xmax_person :
                                without_helmet_detected = True
                                # print("Alert: A person with a helmet detected with 0% score")
                            else:
                                cv2.imshow('Object detector', frame)
                                # k = vis_util.visualize_boxes_and_labels_on_image_array(
                                #     frame,
                                #     np.squeeze(boxes_helmet),
                                #     np.squeeze(classes_helmet).astype(np.int32),
                                #     np.squeeze(scores_helmet),
                                #     category_index_helmet,
                                #     use_normalized_coordinates=True,
                                #     line_thickness=2,
                                #     min_score_thresh=0.0)
                                # out.write(k)
                                # cv2.imshow('Object detector', k)
                                # break  # Exit loop if a helmet is detected inside the person box

                    # Draw a red bounding box if no helmet is detected inside the person box
                        if without_helmet_detected:
                            print("No helmet detected for this person. Drawing red bounding box.")
                            # Write the frame with detections to the output video
                            print("Coordinates of top-left corner:", ((xmin_person), (ymin_person)))
                            print("Coordinates of bottom-right corner:", ((xmax_person), (ymax_person)))
                            start_point = (int(ymin_person * imh), int(xmin_person * imw))
                            end_point = (int(ymax_person * imh), int(xmax_person * imw))
                            # Blue color in BGR
                            color = (0, 0, 255)
                            # Line thickness of 2 px
                            thickness = 2
                            # Using cv2.rectangle() method
                            # Draw a rectangle with blue line borders of thickness of 2 px
                            image = cv2.rectangle(frame, start_point, end_point, color, thickness)
                            # cv2.rectangle(frame,start_point, (int(xmax_person), int(ymax_person)),
                            #               (0, 0, 255), 2)
                            cv2.imshow('Object detector', image)
                            cropped = image[int(ymin_person * imh):int(imh),
                                    int(xmin_person * imw) - 100:int(xmax_person * imw)]
                            cv2.imwrite('cropped_image.jpg', cropped)
                            
                            a,b=detect_number('cropped_image.jpg')
                            if(a):
                                return 1,b,cropped
                            else:
                                return 2,b,cropped
                        

        # Display the processed frame
        # cv2.imshow('Object detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()