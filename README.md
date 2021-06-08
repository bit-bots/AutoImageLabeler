# AutoImageLabeler
AutoImageLabeler creates new predictions on images.
The predictions are output in the upload format required by the [ImageTagger](https://github.com/bit-bots/imagetagger).

Two different methods are available.
1. YOLO (bounding boxes only)
1. Fully Convolutional Neural Networks (polygon output)

Since the used methods can be very different, we will explain them individually:

## YOLO


### Create labels for training
We provide a script for creating YOLO labels ([createYoloLabels.py](https://github.com/bit-bots/AutoImageLabeler/blob/master/yolo/createYoloLabels.py)).
The script currently works for the classes `ball`, `goalpost`, `robot`, `L-Intersection`, `T-Intersection`, `X-Intersection`, and the `crossbar`.
For the intersections, we create a bounding box of 5% of the image height and 5% of the image width.
The labels must be provided in the form of a yaml file.
You can call the script with e.g. `python3 createYoloLabels.py /foo/imagesets`.
The path should be an absolute path.
You do not need to specify the path to the `.yaml` file, the script searches for them in the root directory and all subfolders of the given path.
It assumes the `.yaml` file is in the same folder as the images for which it contains annotations.
The script assumes your images use one of the following fileendings: `.jpg, .JPG, .png, .PNG`

After calling the script, it will tell you which folders it found and the `.yaml` files it found.
It will then create a .txt file for every image where an annotation exists in the .yaml file.
In your root directory a `train.txt` file will be saved.
This file contains the paths to all of your images which should be included in the training.
As absolute paths are used, you probably need to run the script on the machine where the training will happen.
Otherwise, the filepaths might not be correct.
We assume if and only if an annotation (including "not in image") exists for an image, it should be used in the training.

###### TL;DR
`python3 createYoloLabels.py /absolute/path/to/imagesets/`

### When you already have a trained YOLO
We assume that you only need predictions for a bounding box.
If you want to detect e.g. a goalpost, you probably want to use another approach to predict more precisely and make use of the precision a polygon can allow for.

To use this you need to the following steps:
1. put a directory called ``yoloConfig`` into the yolo folder
1. add the required files into this folder (they were used during training/the weights)
    1. the yolo config file as: ``config.cfg``
    1. the names of your objects: ``obj.names``
    1. the trained weights as: ``yolo_weights.weights``
1. change ``directory = "/tmp/imgs"`` to the folder where your images are in
1. call `` python3 generatePredictions.py``
1. Upload the ``output.txt`` that was generated to the ImageTagger

## FCN (Fully Convolutional Neural Network)

This approach can do pixel precise predictions of the objects.
This is superior in potential accuracy compared to the YOLO approach which can only output a bounding box.
For this approach you do not need to have a trained neural network already.

There are two approaches, namely one with pytorch and one with keras.
The approach with pytorch has not been tested yet, so do not rely on it.
The approach with keras has been tested.

To use the keras approach you need to do the following steps:
1. change the parameters in __init__ in ``fcn/keras/dataset.py``
1. train a neural network with ``fcn/keras/train.py`` after changing the parameters.
1. change the ``directory = `` line to where your images to be predicted are located
1. call python3 generatePredictions.py
1. Upload the ``output.txt`` that was generated to the ImageTagger
