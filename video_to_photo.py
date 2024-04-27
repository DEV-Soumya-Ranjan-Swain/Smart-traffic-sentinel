import cv2
import os
import motorbike_helmet_detection_image as m_h

video_path = "test1.mp4"
output_directory = 'data'
time_interval = .6  # Interval in seconds

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Open the video file
cam = cv2.VideoCapture(video_path)
if not cam.isOpened():
    print("Error: Unable to open video file.")
    exit()

current_frame = 0
previous_timestamp = 0

while True:
    ret, frame = cam.read()
    if ret:
        # Get the current timestamp of the frame in seconds
        current_timestamp = cam.get(cv2.CAP_PROP_POS_MSEC) / 1000

        # Check if the current frame's timestamp is at least 2 seconds ahead of the previous frame's timestamp
        if current_timestamp - previous_timestamp >= time_interval:
            # Save the frame
            frame_name = f'{output_directory}/frame{current_frame}.jpg'
            cv2.imwrite(frame_name, frame)
            m_h.check_each_img(frame_name)
            print(f'Frame {current_frame} saved at {current_timestamp:.2f} seconds')

            # Update the previous timestamp and frame counter
            previous_timestamp = current_timestamp
            current_frame += 1

    else:
        break

# Release the video capture object
cam.release()
cv2.destroyAllWindows()
