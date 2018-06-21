from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
from styx_msgs.msg import TrafficLight
import rospy

class TLClassifier(object):
    def __init__(self):
        self.classification_graph = self.load_graph('light_classification/classification_frozen_graph/real_frozen_inference_graph.pb')
        #self.classification_graph = self.load_graph('light_classification/classification_frozen_graph/sim_frozen_inference_graph.pb')

        self.input_image = self.classification_graph.get_tensor_by_name('image_tensor:0')

        self.detection_classes = self.classification_graph.get_tensor_by_name('detection_classes:0')
        self.detection_number = self.classification_graph.get_tensor_by_name('num_detections:0')
        self.detection_scores = self.classification_graph.get_tensor_by_name('detection_scores:0')
        self.detection_boxes = self.classification_graph.get_tensor_by_name('detection_boxes:0')

        self.categorys = {1: {"name": "Red"}, 2: {"name": "Yellow"}, 3: {"name": "Green"}, 4: {"name": "Unkown"}}

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
        (classes, detection, scores, boxes) = self.sess.run([self.detection_classes, self.detection_number,
                                                            self.detection_scores, self.detection_boxes],
                                                            feed_dict={self.input_image: image})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        for score, class_index in zip(scores, classes):
            print(score)
            if score > 0.3:
                class_name = self.categorys[classes[class_index]]['name']
                rospy.logdebug('TLClassifier: Color = %s', class_name)

                if class_name == 'Red':
                    return TrafficLight.RED
                elif class_name == 'Yellow':
                    return TrafficLight.YELLOW
                elif class_name == 'Green':
                    return TrafficLight.GREEN

        return TrafficLight.UNKNOWN

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        np_image_expanded = np.expand_dims(image, axis=0)
        return self.run(np_image_expanded)
