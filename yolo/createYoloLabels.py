import os
import glob
import sys

import yaml

if sys.argv[0] == None:
    directory = input("absolute path to root of your datasets:")
else:
    directory = sys.argv[1]
datasets = [x[0] for x in os.walk(directory)]  # generate a list of all subdirectories (including root directory)
datasets = datasets[1:]  # remove root directory
print("The following datasets will be considered:")
for d in datasets:
    print(d)

for d in datasets:
    yamlfile = glob.glob(f"{d}/*.yaml")
    if len(yamlfile) > 1:
        print(f"There was more than one yaml file in {d}, this is probably unwanted...")
        print("I will use {} now. Be careful if this is not the one you expected me to use.".format(yamlfile[0]))
    with open(yamlfile[0]) as f:
        export = yaml.safe_load(f)


    for name, frame in export['images'].items():
        annolist = []
        for annotation in frame['annotations']:
            if not (annotation['vector'][0] == 'notinimage'):
                imgwidth = frame['width']
                imgheight = frame['height']
                for annotation in frame['annotations']:
                    if not (annotation['vector'][0] == 'notinimage'):
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

                        if annotation['type'] == "robot":
                            classid = 0

                        annolist.append("{} {} {} {} {}".format(classid, relcenter_x, relcenter_y, relannowidth, relannoheight,))
                    else:
                        pass

        imgname = name.replace(".png", "").replace(".jpg", "")
        with open(d + "/" + imgname + ".txt", "w+") as output:
            for e in annolist:
                output.write(e + "\n")


