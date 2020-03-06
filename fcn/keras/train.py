from keras_segmentation.models.unet import vgg_unet

model = vgg_unet(n_classes=2)

model.train(
    train_images = "./labeled/train_images",
    train_annotations = "./labeled/train_segmentation",
    checkpoints_path = "/tmp/net/",
    epochs = 1
)