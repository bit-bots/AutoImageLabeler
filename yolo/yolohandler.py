import cv2
import os
import abc

try:
    from pydarknet import Detector, Image
except ImportError:
    #rospy.logerr(
    #    "Not able to run Darknet YOLO! Its only executable under python3 with yolo34py or yolo34py-gpu installed.",
    #    logger_name="vision_yolo")
    # print("Not able to run Darknet YOLO! Its only executable under python3 with yolo34py or yolo34py-gpu installed.")
    print("Could not import pydarknet. This might be fine if you intend to use CV2.")
import numpy as np


class YoloHandler():
    """
    Defines an abstract YoloHandler
    """

    def __init__(self, model_path):
        """
        Init abstract YoloHandler.
        """
        # self._ball_candidates = None
        # self._goalpost_candidates = None
        self.candidates = None
    @abc.abstractmethod
    def set_image(self, img):
        """
        Image setter abstact method. (Cached)

        :param img: Image
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self):
        """
        Implemented version should run the neural metwork on the latest image. (Cached)
        """
        raise NotImplementedError

    def get_candidates(self):
        """
        Runs neural network and returns results for all classes. (Cached)
        """
        return self.candidates


class YoloHandlerDarknet(YoloHandler):
    """
    Yolo34py library implementation of our yolo model
    """

    def __init__(self, model_path):
        """
        Yolo constructor
        :param model_path: path to the yolo model
        """
        # Define more paths
        weightpath = os.path.join(model_path, "yolo_weights.weights")
        configpath = os.path.join(model_path, "config.cfg")
        datapath = os.path.join("/tmp/obj.data")
        namepath = os.path.join(model_path, "obj.names")
        # Generates a dummy file for the library
        self._generate_dummy_obj_data_file(namepath)

        # Setup detector
        self._net = Detector(bytes(configpath, encoding="utf-8"), bytes(weightpath, encoding="utf-8"), 0.5,
                             bytes(datapath, encoding="utf-8"))
        # Set cached stuff
        self._image = None
        self._results = None
        super(YoloHandlerDarknet, self).__init__(model_path)

    def _generate_dummy_obj_data_file(self, obj_name_path):
        """
        Generates a dummy object data file.
        In which some meta information for the library is stored.

        :param obj_name_path: path to the class name file
        """
        # Generate file content
        obj_data = "classes = 52\nnames = " + obj_name_path
        # Write file
        with open('/tmp/obj.data', 'w') as f:
            f.write(obj_data)

    def set_image(self, image):
        """
        Set a image for yolo. This also resets the caches.

        :param image: current vision image
        """
        # Check if image has been processed
        if np.array_equal(image, self._image):
            return
        # Set image
        self._image = image
        # Reset cached stuff
        self._results = None
        self._goalpost_candidates = None
        self._ball_candidates = None

    def predict(self):
        """
        Runs the neural network
        """
        # Check if cached
        if self._results is None:
            # Run neural network
            self._results = self._net.detect(Image(self._image))
            # Init lists
            self._ball_candidates = []
            self._goalpost_candidates = []
            # Go through results
            for out in self._results:
                # Get class id
                class_id = out[0]
                # Get confidence
                confidence = out[1]
                # Get candidate position and size
                x, y, w, h = out[2]
                x = x - int(w // 2)
                y = y - int(h // 2)
                # Create candidate
                c = Candidate(int(x), int(y), int(w), int(h), confidence)
                # Append candidate to the right list depending on the class
                if class_id == b"ball":
                    self._ball_candidates.append(c)
                if class_id == b"goalpost":
                    self._goalpost_candidates.append(c)


class YoloHandlerOpenCV(YoloHandler):
    """
    Opencv library implementation of our yolo model
    """

    def __init__(self, model_path):
        # Build paths
        weightpath = os.path.join(model_path, "yolo_weights.weights")
        configpath = os.path.join(model_path, "config.cfg")
        # Set config
        #self._config = config
        # Settings
        self._nms_threshold = 0.4
        self._confidence_threshold = 0.5
        # Setup neural network
        self._net = cv2.dnn.readNet(weightpath, configpath)
        # Set default state to all cached values
        self._image = None
        self._blob = None
        self._outs = None
        self._results = None
        super(YoloHandlerOpenCV, self).__init__(model_path)

    def _get_output_layers(self):
        """
        Library stuff
        """
        layer_names = self._net.getLayerNames()

        output_layers = [layer_names[i[0] - 1] for i in self._net.getUnconnectedOutLayers()]

        return output_layers

    def set_image(self, image):
        """
        Set a image for yolo. This also resets the caches.

        :param image: current vision image
        """
        # Check if image has been processed
        if np.array_equal(image, self._image):
            return
        # Set image
        self._image = image
        self._width = image.shape[1]
        self._height = image.shape[0]
        # Create blob
        self._blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        # Reset cached stuff
        self._outs = None
        self._goalpost_candidates = None
        self._ball_candidates = None

    def predict(self):
        """
        Runs the neural network
        """
        # Check if cached
        if self._outs is None:
            # Set image
            self._net.setInput(self._blob)
            # Run net
            self._outs = self._net.forward(self._get_output_layers())
            # Create lists
            class_ids = []
            confidences = []
            boxes = []
            self.candidates = []
            # Iterate over output/detections
            for out in self._outs:
                for detection in out:
                    # Get score
                    scores = detection[5:]
                    # Get class
                    class_id = np.argmax(scores)
                    # Get confidence from score
                    confidence = scores[class_id]
                    # Static threshold
                    if confidence > 0.5:
                        # Get center point of the candidate
                        center_x = int(detection[0] * self._width)
                        center_y = int(detection[1] * self._height)
                        # Get the heigh/width
                        w = int(detection[2] * self._width)
                        h = int(detection[3] * self._height)
                        # Calc the upper left point
                        x = center_x - w / 2
                        y = center_y - h / 2
                        # Append result
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            # Merge boxes
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self._confidence_threshold, self._nms_threshold)

            # Iterate over filtered boxes
            for i in indices:
                # Get id
                i = i[0]
                # Get box
                box = boxes[i]
                # Convert the box position/size to int
                x = int(box[0])
                y = int(box[1])
                w = int(box[2])
                h = int(box[3])
                # Append candidate to the right list
                class_id = class_ids[i]
                self.candidates.append([x,y,w,h,confidences[i],class_id])