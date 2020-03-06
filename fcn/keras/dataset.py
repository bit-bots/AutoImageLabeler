from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
from pathlib import Path
import copy
import cv2
import numpy as np
import torch
from pathlib import Path
import os

class roboCupDatasets(Dataset):

    def __init__(self, rootDataDirectory = "/srv/ssd_nvm/15hagge/data/", outputDirectory= "/srv/ssd_nvm/15hagge/labeled", labelsToUse = ["goalpost"]):
        """
        Args:
            rootDataDirectory (string): path to directory of directory with images
                                        (can also be path to single directory)
            outputDirectory (string): path to directory where images will be stored with their labels
            labelsToUse ([string]): list of labels that should be learned
        """
        self.imagelist = list()
        self.labelFiles = list()

        # create output directory
        outputImages = outputDirectory + "/train_images/"
        Path(outputImages).mkdir(parents=True, exist_ok=True)
        outputSegmentations = outputDirectory + "/train_segmentation/"
        Path(outputSegmentations).mkdir(parents=True, exist_ok=True)

        for filename in Path(rootDataDirectory).rglob("*.yaml"):
            self.labelFiles.append(filename)

        for path in self.labelFiles:
            with open(path, "r") as f:
                annos = yaml.safe_load(f)
                images = annos["images"]
                for image in images.items():
                    # get parent path of yaml and add image path
                    # this assumes the .yaml file is in the same folder as the images
                    imagepath = str(path.parent.absolute()) + "/" + image[0]
                    readImg = cv2.imread(imagepath)
                    height, width, channels = readImg.shape
                    # create black image with same dimensions as input image
                    label = np.zeros((height,width, channels)).astype('uint8')
                    labelFound = False
                    for annotation in image[1]["annotations"]:
                        if annotation["type"] in labelsToUse:
                            labelFound = True
                            polygon = annotation["vector"]
                            if not "notinimage" in polygon:
                                cv2.fillConvexPoly(label, np.array(polygon), [1,1,1])
                    # only use images where we have labels available (notinimage is a label too)
                    if not labelFound:
                        continue
                    # Debug Labels:
                    # cv2.imshow("label", label)
                    # cv2.waitKey()
                    # define imagename as imageset+name. Otherwise images in multiple sets could have the same name
                    imageset = imagepath.split("/")[-2:]
                    imagename = imageset[0] + imageset [1]

                    cv2.imwrite(outputImages + imagename, readImg)
                    # multiplying by 255 makes the labels visible on the output images
                    # label = label * 255
                    labelname, _ = os.path.splitext(imagename)
                    labelname = labelname + ".png"
                    cv2.imwrite(outputSegmentations + labelname, label)
                    # self.imagelist.append((imagepath, label))

if __name__ == "__main__":
    a = roboCupDatasets()

