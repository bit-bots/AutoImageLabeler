# AutoImageLabeler
AutoImageLabeler creates new predictions on images.
The predictions are output in the upload format required by the [ImageTagger](https://github.com/bit-bots/imagetagger).

Two different methods are available.
1. YOLO (bounding boxes only)
1. Fully Convolutional Neural Networks (polygon output)

Since the used methods can be very different, we will explain them individually:

YOLO
----

For the YOLO method it is assumed you already have a trained YOLO that can detect your objects.
We also assume that you only need predictions for a bounding box.
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

FCN (Fully Convolutional Neural Network)
----------------------------------------

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
