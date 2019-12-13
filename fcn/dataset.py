from torch.utils.data.dataset import Dataset
from torchvision import transforms
import yaml
from pathlib import Path
import copy
import cv2
import numpy as np
import torch

class roboCupDatasets(Dataset):

    def __init__(self, rootDataDirectory = "../data/", labelsToUse = ["goalpost"], transform=None):
        """
        Args:
            rootDataDirectory (string): path to directory of directory with images
                                        (can also be path to single directory)
            transform (callable, optional): transforms that should be applied to images
        """
        self.imagelist = list()
        self.labelFiles = list()
        self.transform = transform

        for filename in Path(rootDataDirectory).rglob("*.yaml"):
            self.labelFiles.append(filename)

        path = self.labelFiles[0]
        with open(path, "r") as f:
            annos = yaml.safe_load(f)
            images = annos["images"]
            for image in images.items():
                # get parent path of yaml and add image path
                # this assumes the .yaml file is in the same folder as the images
                imagepath = str(path.parent.absolute()) + "/" + image[0]
                readImg = cv2.imread(imagepath)
                height, width, _ = readImg.shape
                # create black image with same dimensions as input image
                label = np.zeros((height,width))
                labelFound = False
                for annotation in image[1]["annotations"]:
                    if annotation["type"] in labelsToUse:
                        labelFound = True
                        polygon = annotation["vector"]
                        if not "notinimage" in polygon:
                            cv2.fillConvexPoly(label, np.array(polygon), 1.0)
                # only use images where we have labels available
                if not labelFound:
                    continue
                # Debug Labels:
                #cv2.imshow("label", label)
                #cv2.waitKey()
                # store imagepath instead of image to be more memoryefficient
                self.imagelist.append((imagepath, label))

    def __getitem__(self, index):
        # TODO apply transforms of choice
        # todo transform to tensor
        img = cv2.imread(self.imagelist[index][0])
        label = self.imagelist[index][1]
        # assuming this is an fcnn, the same transformation can be applied to the label...
        # unless the transformation does stuff with colors...
        # todo figure out which transformations to apply to label too
        if self.transform:
            img = self.transform(img)
            label = self.transform(label)
        # transform to tensor
        # numpy image: H x W x C
        # torch image: C X H X W
        #print(img.shape)
        #img = img.transpose((2, 0, 1))
        #label = label.transpose((2, 0, 1))
        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self):
        len(self.imagelist)

if __name__ == "__main__":
    a = roboCupDatasets()
    img = a.__getitem__(0)[0]
    print(img)


