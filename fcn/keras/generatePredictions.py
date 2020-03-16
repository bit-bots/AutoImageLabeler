import cv2
import numpy as np
from keras_segmentation.models.unet import mobilenet_unet
from pathlib import Path

directory = "insert path here"
imagelist = []

annoType = "goalpost"
model = mobilenet_unet(n_classes = 2)
model.load_weights("./mobile.68")

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

    # the following taken from https://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html (7.b)
    ret, thresh = cv2.threshold(prediction, 155, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    vectorlist = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        #cv2.drawContours(prediction, [box], 0, (0, 0, 255), 2)
        vector = f"""{{"x1": {box[0][0]}, "y1": {box[0][1]}, "x2": {box[1][0]}, "y2": {box[1][1]},"""\
                 f""""x3": {box[2][0]}, "y3": {box[2][1]}, "x4": {box[3][0]}, "y4": {box[3][1]}}}"""
        vectorlist.append(vector)
        print(vector)

    #cv2.imshow("foo", prediction)
    #cv2.waitKey(0)

    imagename = image.name
    with open("output.txt", "a") as f:
        for candidate in vectorlist:
            imagetaggerformat = f"{imagename}|{annoType}|{candidate}|\n"
            f.write(imagetaggerformat)


