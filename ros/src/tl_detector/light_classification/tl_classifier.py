from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        self.classification_graph = self.load_graph('light_classification/frozen_classification_graph.pb')
        self.input_image = self.classification_graph.get_tensor_by_name('image_tensor:0')

        self.detection_classes = self.classification_graph.get_tensor_by_name('detection_classes:0')
        self.detection_number = self.classification_graph.get_tensor_by_name('num_detections:0')
        self.detection_scores = self.classification_graph.get_tensor_by_name('detection_scores:0')
        self.detection_boxes = self.classification_graph.get_tensor_by_name('detection_boxes:0')

        self.sess = tf.Session(graph=self.classification_graph)

    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        return graph

    def run(self, image):
        classes, detection, scores, boxes = self.sess.run([self.detection_classes, self.detection_number,
                                                           self.detection_scores, self.detection_boxes],
                                                          feed_dict={self.input_image: image})

        return self.resolve_traffic_light(int(np.squeeze(classes)[0]))

    def resolve_traffic_light(self, classification):
        switcher = {
            1: TrafficLight.GREEN,
            2: TrafficLight.RED,
            3: TrafficLight.GREEN,
            4: TrafficLight.GREEN,
            5: TrafficLight.RED,
            6: TrafficLight.RED,
            7: TrafficLight.YELLOW,
            8: TrafficLight.UNKNOWN,
            9: TrafficLight.RED,
            10: TrafficLight.GREEN,
            11: TrafficLight.GREEN,
            12: TrafficLight.GREEN,
            13: TrafficLight.RED,
            14: TrafficLight.RED
        }

        return switcher.get(classification, TrafficLight.UNKNOWN)

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        np_image = self.load_image_into_numpy_array(image)
        np_image_expanded = np.expand_dims(np_image, axis=0)
        return self.run(np_image_expanded)
