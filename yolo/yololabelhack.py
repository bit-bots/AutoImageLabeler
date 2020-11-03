import yaml
import glob

yamlfile = glob.glob("*.yaml")
if len(yamlfile) > 1:
    print("There was more than one yaml file in this directory, this is probably unwanted...")
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
    with open(imgname + ".txt", "w+") as output:
        for e in annolist:
            output.write(e + "\n")


