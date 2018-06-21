from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
from styx_msgs.msg import TrafficLight
import rospy

class TLClassifier(object):
    def __init__(self):
        self.classification_graph = self.load_graph('light_classification/mobilenet_frozen_graph/real_frozen_inference_graph.pb')
        #self.classification_graph = self.load_graph('light_classification/mobilenet_frozen_graph/sim_frozen_inference_graph.pb')

        self.input_image = self.classification_graph.get_tensor_by_name('image_tensor:0')

        self.detection_classes = self.classification_graph.get_tensor_by_name('detection_classes:0')
        self.detection_number = self.classification_graph.get_tensor_by_name('num_detections:0')
        self.detection_scores = self.classification_graph.get_tensor_by_name('detection_scores:0')
        self.detection_boxes = self.classification_graph.get_tensor_by_name('detection_boxes:0')

        self.sess = tf.Session(graph=self.classification_graph)

    def load_graph(self, graph_file):
        rospy.loginfo('Loading Graph_file "' + graph_file + '""')
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        rospy.loginfo('Graph_file loaded.')
        return graph

    def run(self, image):
        rospy.loginfo('TLClassifier.run()...')
        classes, detection, scores, boxes = self.sess.run([self.detection_classes, self.detection_number,
                                                           self.detection_scores, self.detection_boxes],
                                                          feed_dict={self.input_image: image})

        detected_class = int(np.squeeze(classes)[0])
        traffic_light = self.resolve_traffic_light(detected_class)
        rospy.loginfo('Traffic light: ' + self.resolve_traffic_light_text(detected_class))
        return traffic_light

    def resolve_traffic_light(self, classification):
        switcher = {
            1: TrafficLight.GREEN,
            2: TrafficLight.RED,
            3: TrafficLight.YELLOW,
            4: TrafficLight.UNKNOWN
        }
        return switcher.get(classification, TrafficLight.UNKNOWN)

    def resolve_traffic_light_text(self, classification):
        switcher = {
            1: "GREEN",
            2: "RED",
            3: "YELLOW",
            4: "UNKNOWN"
        }
        return switcher.get(classification, "UNKNOWN")

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        np_image_expanded = np.expand_dims(image, axis=0)
        return self.run(np_image_expanded)