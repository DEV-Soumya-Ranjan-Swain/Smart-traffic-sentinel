import os
import cv2
import numpy as np
import tensorflow as tf
import sys

from playsound import playsound

from utils import label_map_util
from utils import visualization_utils as vis_util

detection_graph = tf.Graph()


# Import utilites
# from utils import label_map_util

# Name of the directory containing the object detection module we're using
TRAINED_MODEL_DIR = 'frozen_graphs'

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = 'frozen_inference_graph_motorbike.pb'
print(PATH_TO_CKPT)
# Path to label map file
PATH_TO_LABELS = 'labelmap_motorbike.pbtxt'

NUM_CLASSES = 1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

print(category_index)

print("> ====== Loading frozen graph into memory")
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)
    print(">  ====== Inference graph loaded.")

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')


# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

print(image_tensor)
detection_boxes
image = cv2.imread('images4.jpeg')
image.size

image = cv2.resize(image, (800,700))
image_expanded = np.expand_dims(image, axis=0)
print(image.size)
print(image.shape)
class_names_mapping = {
            1: "with_helmet",2:"motorcycle",3:"without_helmet"}

(boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
result = scores.flatten()
res = []
for idx in range(0, len(result)):
    if result[idx] > .40:
        res.append(idx)

top_classes = classes.flatten()
# Selecting class 2 and 3
#top_classes = top_classes[top_classes > 1]
res_list = [top_classes[i] for i in res]

class_final_names = [class_names_mapping[x] for x in res_list]
top_scores = [e for l2 in scores for e in l2 if e > 0.30]

scores.flatten()
new_scores = scores.flatten()

new_boxes = boxes.reshape(100, 4)

# get all boxes from an array
max_boxes_to_draw = new_boxes.shape[0]
# this is set as a default but feel free to adjust it to your needs
min_score_thresh = .30
vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=6,
    min_score_thresh=0.30)
cv2.imshow('Object detector', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Object detector', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
result = scores.flatten()
res = []
for idx in range(0, len(result)):
    if result[idx] > .40:
        res.append(idx)

top_classes = classes.flatten()
# Selecting class 2 and 3
#top_classes = top_classes[top_classes > 1]
res_list = [top_classes[i] for i in res]

class_final_names = [class_names_mapping[x] for x in res_list]
top_scores = [e for l2 in scores for e in l2 if e > 0.30]


vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.75)
    # All the results have been drawn on the frame, so it's time to display it.
cv2.imshow('Object detector', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

result = scores.flatten()
res = []
for idx in range(0, len(result)):
    if result[idx] > .40:
        res.append(idx)

k=np.squeeze(classes).astype(np.int32).flatten

k