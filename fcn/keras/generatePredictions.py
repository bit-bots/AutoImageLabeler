import cv2
import numpy as np
from keras_segmentation.models.unet import mobilenet_unet
import keras_segmentation
from pathlib import Path
import math

directory = "/srv/ssd_nvm/15hagge/testset/375/"
imagelist = []

annoType = "robot"
model = mobilenet_unet(n_classes = 2)
model.load_weights("nets/robots/mobile.69")

for filename in Path(directory).rglob("*.png"):
    imagelist.append(filename)
for filename in Path(directory).rglob("*.jpg"):
    imagelist.append(filename)
print(f"Found {len(imagelist)} images in {directory}")

for index, image in enumerate(imagelist):
    # \r and end="" so the same line is used again
    print(f"\rpredicting for image {index+1}/{len(imagelist)}", end="")
    img = cv2.imread(str(image))
    print(str(image))
    prediction = model.predict_segmentation(inp=img, out_fname="/tmp/stuff.png")
    prediction = cv2.imread("/tmp/stuff.png", cv2.IMREAD_GRAYSCALE)

    # the following in part taken from https://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html (7.b)
    # everything below 155 seems to be background. 
    # 210 makes for better fitting bounding boxes on one test set. Might have to evaluate further
    ret, thresh = cv2.threshold(prediction, 210, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    vectorlist = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # skip too small boxes. It is unlikely this is an actual goalpost
        if math.sqrt((box[0][0] - box[2][0]) ** 2 + (box[0][1] - box[2][1]) ** 2) < 50:
            continue
        # cv2.drawContours(prediction, [box], 0, (0, 0, 255), 2)
        # 4 coordinates to describe object
        if annoType == "goalpost":
            vector = f"""{{"x1": {box[0][0]}, "y1": {box[0][1]}, "x2": {box[1][0]}, "y2": {box[1][1]},"""\
                     f""""x3": {box[2][0]}, "y3": {box[2][1]}, "x4": {box[3][0]}, "y4": {box[3][1]}}}"""
        # bounding box
        elif annoType == "robot":
            minx = 100000
            miny = 100000
            maxx = 0
            maxy = 0
            # get min / max values for both axis to determine bounding box
            for coordinate in box:
                if coordinate[0] < minx:
                    minx = coordinate[0]
                if coordinate[0] > maxx:
                    maxx = coordinate[0]
                if coordinate[1] < miny:
                    miny = coordinate[1]
                if coordinate[1] > maxy:
                    maxy = coordinate[1]
            vector = f"""{{"x1": {minx}, "y1": {miny}, "x2": {maxx}, "y2": {maxy}}}"""
        else:
            print("unknown label type")
        vectorlist.append(vector)
        # print(vector)

    #cv2.imshow("img", img)
    #cv2.imshow("prediction", prediction)
    #cv2.waitKey(0)

    imagename = image.name
    with open("output.txt", "a") as f:
        for candidate in vectorlist:
            imagetaggerformat = f"{imagename}|{annoType}|{candidate}|\n"
            f.write(imagetaggerformat)


