import cv2
import numpy as np
from keras_segmentation.models.unet import mobilenet_unet
from pathlib import Path

directory = "insert path here"
imagelist = []


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
    prediction = model.predict_segmentation(inp=img)
    prediction = cv2.imread(prediction, cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(prediction, 155, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(prediction, [box], 0, (0, 0, 255), 2)
        #print(box)

    cv2.imshow("foo", prediction)
    cv2.waitKey(0)


'''
im = cv2.imread("/home/jonas/Downloads/bar.png", cv2.IMREAD_GRAYSCALE)

ret, thresh = cv2.threshold(im, 155, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(im,[box],0,(0,0,255),2)
    print(box)


cv2.imshow("foo", im)
cv2.waitKey(0)
'''