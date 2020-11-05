from pathlib import Path
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


class Candidate:
    """
    A :class:`.Candidate` is a representation of an arbitrary object in an image.
    It is very similar to bounding boxes but with an additional rating.
    This class provides several getters for different properties of the candidate.
    """
    def __init__(self, x1=0, y1=0, width=0, height=0, rating=None):
        """
        Initialization of :class:`.Candidate`.
        :param int x1: Horizontal part of the coordinate of the top left corner of the candidate
        :param int y1: Vertical part of the coordinate of the top left corner of the candidate
        :param int width: Horizontal size
        :param int height: Vertical size
        :param float rating: Confidence of the candidate
        """
        self._x1 = x1
        self._y1 = y1
        self._width = width
        self._height = height
        self._rating = rating

    def get_width(self):
        # type: () -> int
        """
        :return int: Width of the candidate bounding box.
        """
        return self._width

    def get_height(self):
        # type: () -> int
        """
        :return int: Height of the candidate bounding box.
        """
        return self._height

    def get_center_x(self):
        # type: () -> int
        """
        :return int: Center x coordinate of the candidate bounding box.
        """
        return self._x1 + int(self._width // 2)

    def get_center_y(self):
        # type: () -> int
        """
        :return int: Center y coordinate of the candidate bounding box.
        """
        return self._y1 + int(self._height // 2)

    def get_center_point(self):
        # type: () -> tuple[int, int]
        """
        :return tuple[int,int]: Center point of the bounding box.
        """
        return self.get_center_x(), self.get_center_y()

    def get_diameter(self):
        # type: () -> int
        """
        :return int: Mean diameter of the candidate.
        """
        return int((self._height + self._width) // 2)

    def get_radius(self):
        # type: () -> int
        """
        :return int: Mean radius of the candidate.
        """
        return int(self.get_diameter() // 2)

    def get_upper_left_point(self):
        # type: () -> tuple[int, int]
        """
        :return tuple[int,int]: Upper left point of the candidate.
        """
        return self._x1, self._y1

    def get_upper_left_x(self):
        # type: () -> int
        """
        :return int: Upper left x coordinate of the candidate.
        """
        return self._x1

    def get_upper_left_y(self):
        # type: () -> int
        """
        :return int: Upper left y coordinate of the candidate.
        """
        return self._y1

    def get_lower_right_point(self):
        # type: () -> tuple[int, int]
        """
        :return tuple[int,int]: Lower right point of the candidate.
        """
        return self._x1 + self._width, self._y1 + self._height

    def get_lower_right_x(self):
        # type: () -> int
        """
        :return int: Lower right x coordinate of the candidate.
        """
        return self._x1 + self._width

    def get_lower_right_y(self):
        # type: () -> int
        """
        :return int: Lower right y coordinate of the candidate.
        """
        return self._y1 + self._height

    def get_lower_center_point(self):
        # type: () -> (int, int)
        """
        :return tuple: Returns the lowest point of the candidate. The point is horizontally centered inside the candidate.
        """
        return (self.get_center_x(), self.get_lower_right_y())

    def set_rating(self, rating):
        # type: (float) -> None
        """
        :param float rating: Rating to set.
        """
        if self._rating is not None:
            rospy.logwarn('Candidate rating has already been set.', logger_name='Candidate')
            return
        self._rating = rating

    def get_rating(self):
        # type: () -> float
        """
        :return float: Rating of the candidate
        """
        return self._rating

    def point_in_candidate(self, point):
        # type: (tuple) -> bool
        """
        Returns whether the point is in the candidate or not.
        In the process, the candidate gets treated as a rectangle.
        :param point: An x-y-int-tuple defining the point to inspect.
        :return bool: Whether the point is in the candidate or not.
        """
        return (
                self.get_upper_left_x()
                <= point[0]
                <= self.get_upper_left_x() + self.get_width()) \
            and (
                self.get_upper_left_y()
                <= point[1]
                <= self.get_upper_left_y() + self.get_height())

    def get_mAP(self):
        return f"robot {self.get_rating()} {self.get_upper_left_x()} {self.get_upper_left_y()} {self.get_lower_right_x()} {self.get_lower_right_y()}"

    @staticmethod
    def sort_candidates(candidatelist):
        """
        Returns a sorted list of the candidates.
        The first list element is the highest rated candidate.
        :param [Candidate] candidatelist: List of candidates
        :return: List of candidates sorted by rating, in descending order
        """
        return sorted(candidatelist, key = lambda candidate: candidate.get_rating(), reverse=True)

    @staticmethod
    def select_top_candidate(candidatelist):
        """
        Returns the highest rated candidate.
        :param candidatelist: List of candidates
        :return Candidate: Top candidate
        """
        if candidatelist:
            return Candidate.sort_candidates(candidatelist)[0]
        else:
            return None

    @staticmethod
    def rating_threshold(candidatelist, threshold):
        """
        Returns list of all candidates with rating above given threshold.
        
        :param [Candidate] candidatelist: List of candidates to filter
        :param float threshold: Filter threshold
        :return [Candidate]: Filtered list of candidates
        """
        return [candidate for candidate in candidatelist if candidate.get_rating() > threshold]

    def __str__(self):
        """
        Returns string representation of candidate.
        
        :return str: String represeatation of candidate
        """
        return 'x1,y1: {0},{1} | width,height: {2},{3} | rating: {4}'.format(
            self.get_upper_left_x(),
            self.get_upper_left_y(),
            self.get_width(),
            self.get_height(),
            self._rating)


class CandidateFinder(object):
    """
    The abstract class :class:`.CandidateFinder` requires its subclasses to implement the methods
    :meth:`.get_candidates` and :meth:`.compute`.
    Examples of such subclasses are :class:`bitbots_vision.vision_modules.obstcle.ObstacleDetector` and
    :class:`bibtots_vision.vision_modules.fcnn_handler.FcnnHandler`.
    They produce a set of so called *Candidates* which are instances of the class :class:`bitbots_vision.vision_modules.candidate.Candidate`.
    """
    def __init__(self):
        """
        Initialization of :class:`.CandidateFinder`.
        """
        super(CandidateFinder, self).__init__()

    def get_top_candidates(self, count=1):
        """
        Returns the count highest rated candidates.
        :param int count: Number of top-candidates to return
        :return [Candidate]: The count top-candidates
        """
        candidates = self.get_candidates()
        candidates = Candidate.sort_candidates(candidates)
        return candidates[:count]

    def get_top_candidate(self):
        """
        Returns the highest rated candidate.
        :return Candidate: Top candidate or None
        """
        return Candidate.select_top_candidate(self.get_candidates())

    @abc.abstractmethod
    def get_candidates(self):
        """
        Returns a list of all candidates.
        :return [Candidate]: Candidates
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute(self):
        """
        Runs the most intense calculation without returning any output and caches the result.
        """
        raise NotImplementedError


class DummyCandidateFinder(CandidateFinder):
    """
    Dummy candidate detector that is used to run the vision pipeline without a neural network e.g. to save computation time for debugging.
    This implementation returns an empty set of candidates and thus replaces the ordinary detection.
    """
    def __init__(self):
        """
        Initialization of :class:`.DummyCandidateFinder`.
        """
        self._detected_candidates = []
        self._sorted_candidates = []
        self._top_candidate = None

    def set_image(self, image):
        """
        Method to satisfy the interface.
        Actually does nothing.
        :param image: current vision image
        """
        pass

    def compute(self):
        """
        Method to satisfy the interface.
        Actually does nothing, except the extrem complicated command 'pass'.
        """
        pass

    def get_candidates(self):
        """
        Method to satisfy the interface.
        Actually does something. It returns an empty list.
        :return: a empty list
        """
        return self._detected_candidates

    def get_top_candidates(self, count=1):
        """
        Method to satisfy the interface.
        It returns an empty list.
        :param count: how many of zero top candidates do you want?
        :return: a empty list
        """
        return self._sorted_candidates

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


class YoloHandlerOpenCV(YoloHandler):
    """
    Opencv library implementation of our yolo model
    """

    def __init__(self, model_path):
        # Build paths
        weightpath = os.path.join(model_path, "yolo_weights.weights")
        configpath = os.path.join(model_path, "config.cfg")
        # Set config
        # self._config = config
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
        self._robot_candidates = None
        self.candidates = []

    def predict(self):
        """
        Runs the neural network
        """
        self._net.setInput(self._blob)
        # Run net
        self._outs = self._net.forward(self._get_output_layers())
        # Create lists
        class_ids = []
        confidences = []
        boxes = []
        self._robot_candidates = []
        # Iterate over output/detections
        for out in self._outs:
            for detection in out:
                # Get score
                scores = detection[5:]
                # Ger class
                class_id = np.argmax(scores)
                # Get confidence from score
                confidence = scores[class_id]
                # First threshold to decrease candidate count and inscrease performance
                if confidence > self._confidence_threshold:
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
            box = list(map(int, box))

            self.candidates.append(Candidate(*box, confidences[i]))


# Todo don't hardcode this
directory = "/tmp/imgs"
imagelist = []

# define the used labels
# It is important to use the same order for the objects as in your .names file!
labels = []
labels.append("robot")

# TODO find png and jpg in one command
for filename in Path(directory).rglob("*.png"):
    imagelist.append(filename)
for filename in Path(directory).rglob("*.jpg"):
    imagelist.append(filename)
print(f"Found {len(imagelist)} images in {directory}")

count = 0
for index, image in enumerate(imagelist):
    # \r and end="" so the same line is used again
    print(f"\rpredicting for image {index+1}/{len(imagelist)}", end="")
    img = cv2.imread(str(image))
    yolo = YoloHandlerOpenCV("yoloConfig")
    yolo.set_image(img)
    yolo.predict()
    result = yolo.get_candidates()

    for candidate in result:
        imgname = os.path.splitext(image)[0]
        with open(imgname + ".txt", "w+") as output:
                output.write(candidate.get_mAP() + "\n")

    print(len(result))
    if count < 20 and len(result) > 0:
        for candidate in result:
            cv2.rectangle(img, candidate.get_upper_left_point(), candidate.get_lower_right_point(), (255, 0, 0), 2)
            # cv2.imshow("robot", img)
            # k = cv2.waitKey(0)  # 0==wait forever
        cv2.imwrite(f"debug{count}.png", img)
        count += 1
