import os
import glob
import sys

import yaml

if len(sys.argv[0]) == 0:
    directory = input("absolute path to root of your datasets:")
else:
    directory = sys.argv[1]
datasets = [x[0] for x in os.walk(directory)]  # generate a list of all subdirectories (including root directory)
datasets = datasets[1:]  # remove root directory
print("The following datasets will be considered:")
for d in datasets:
    print(d)

trainImages = []  # this ensures only images with labels are used
line_intersections = ["L-Intersection", "T-Intersection", "X-Intersection"]

for d in datasets:
    yamlfile = glob.glob(f"{d}/*.yaml")
    if len(yamlfile) > 1:
        print(f"There was more than one yaml file in {d}, this is probably unwanted...")
        print("I will use {} now. Be careful if this is not the one you expected me to use.".format(yamlfile[0]))
    with open(yamlfile[0]) as f:
        export = yaml.safe_load(f)

    for name, frame in export['images'].items():
        trainImages.append(f"{d}/{name}")
        annolist = []
        for annotation in frame['annotations']:
            if not (annotation['vector'][0] == 'notinimage'):
                imgwidth = frame['width']
                imgheight = frame['height']
                if not (annotation['vector'][0] == 'notinimage'):
                    if annotation['type'] not in line_intersections:
                        min_x = min(map(lambda x: x[0], annotation['vector']))
                        max_x = max(map(lambda x: x[0], annotation['vector']))
                        min_y = min(map(lambda x: x[1], annotation['vector']))
                        max_y = max(map(lambda x: x[1], annotation['vector']))

                        annowidth = max_x - min_x
                        annoheight = max_y - min_y
                        relannowidth = annowidth / imgwidth
                        relannoheight = annoheight / imgheight

                        center_x = min_x + (annowidth / 2)
                        center_y = min_y + (annoheight / 2)
                        relcenter_x = center_x / imgwidth
                        relcenter_y = center_y / imgheight
                    else:
                        # line intersections are only a single coordinate
                        # we need a bounding box though
                        # so we assume the point is in the middle
                        # and then make a box of 5% of the image in all directions
                        coords = annotation['vector'][0]
                        relcenter_x = coords[0] / imgwidth
                        relcenter_y = coords[1] / imgheight
                        relannowidth = 0.05
                        relannoheight = 0.05


                    # TODO this needs to be changed from hand for now
                    if annotation['type'] == "ball":
                        classid = 0
                    elif annotation['type'] == "goalpost":
                        classid = 1
                    elif annotation['type'] == "robot":
                        classid = 2
                    elif annotation['type'] == "L-Intersection":
                        classid = 3
                    elif annotation['type'] == "T-Intersection":
                        classid = 4
                    elif annotation['type'] == "X-Intersection":
                        classid = 5
                    else:
                        print(f"Unknown Annotation Type: {annotation['type']}")


                    annolist.append("{} {} {} {} {}".format(classid, relcenter_x, relcenter_y, relannowidth, relannoheight,))
                else:
                    pass

        imgname = name.replace(".png", "").replace(".jpg", "")
        with open(d + "/" + imgname + ".txt", "w+") as output:
            for e in annolist:
                output.write(e + "\n")

trainImages = set(trainImages) # prevent images from showing up twice
with open(f"{directory}/train.txt", "w") as traintxt:
    for e in trainImages:
        traintxt.write(e + "\n")

