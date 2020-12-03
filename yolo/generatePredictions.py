import sys
from pathlib import Path
import cv2
import yolohandler

# Todo don't hardcode this
directory = "/tmp/imgs"
imagelist = []

# define the used labels
labels = []
with open("yoloConfig/obj.names") as f:
    for line in f:
      labels.append(line.strip())

for filename in Path(directory).rglob("*.png"):
    imagelist.append(filename)
for filename in Path(directory).rglob("*.jpg"):
    imagelist.append(filename)
print(f"Found {len(imagelist)} images in {directory}")

# if yolo34py installation doesn't work for some reason, this is here as fallback, comment in if needed
yolo = yolohandler.YoloHandlerOpenCV("yoloConfig")
# yolo = yolohandler.YoloHandlerDarknet("yoloConfig")
for index, image in enumerate(imagelist):
    # \r and end="" so the same line is used again
    print(f"\rpredicting for image {index+1}/{len(imagelist)}", end="")
    img = cv2.imread(str(image))

    yolo.set_image(image = img)
    yolo.predict()
    result = yolo.get_candidates()

    # We need to handle different annotations differently
    # For balls we only need two coordinates
    # Meanwhile goalposts require four
    vectorlist = []
    for candidate in result:
        x = candidate[0]
        y = candidate[1]
        width = candidate[2]
        height = candidate[3]

        if labels[candidate[5]] == "ball" or labels[candidate[5]] == "robot":
            vector = f"""{{"x1": {x}, "y1": {y}, "x2": {x + width}, "y2": {y + height}}}"""
        elif labels[candidate[5]] == "goalpost":
            # ignore goalposts for now, since the yolo bounding box is inaccurate if it is a tilted goalpost
            continue
            vector = f"""{{"x1": {x}, "y1": {y}, "x2": {x + width}, "y2": {y},""" \
                     f""""x3": {x+width}, "y3": {y+height}, "x4": {x}, "y4": {y+height}}}"""
        else:
            print("An unknown annotation type was used. You might want to check what happened")
        vectorlist.append((candidate[5], vector))

    imagename = image.name
    # TODO delete two next lines?
    annoType = None
    vector = None

    # vector format:
    # 'x1': 300,'y1': 300

    # full anno format:
    #  %%imagename|%%type|{%%vector}|%%ifblurredb%%endif%%ifconcealedc%%endif

    # negative annos:
    #  image name|annotation type|not in image
    # TODO have one combined file write instead of constant opening and closing the file
    with open("output.txt", "a") as f:
        for candidate in vectorlist:
            imagetaggerformat = f"{imagename}|{labels[candidate[0]]}|{candidate[1]}|\n"
            f.write(imagetaggerformat)
