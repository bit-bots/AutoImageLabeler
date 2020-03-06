from keras_segmentation.models.unet import mobilenet_unet
from keras_segmentation import train

model = mobilenet_unet(n_classes=2)
'''
model.train(
    train_images = "/srv/ssd_nvm/15hagge/labeled/train_images/",
    train_annotations = "/srv/ssd_nvm/15hagge/labeled/train_segmentation/",
    checkpoints_path = "/tmp/net/mobile",
    epochs = 175,
)
'''

train.train(model,
    train_images = "/srv/ssd_nvm/15hagge/labeled/train_images/",
    train_annotations = "/srv/ssd_nvm/15hagge/labeled/train_segmentation/",
    checkpoints_path = "/tmp/net/mobile",
    epochs = 20,
    #ignore_zero_class = True ,
    gen_use_multiprocessing = True ,
    batch_size = 15
)
