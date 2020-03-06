from keras_segmentation.models.segnet import mobilenet_segnet

model = mobilenet_segnet(n_classes=2)

model.train(
    train_images = "/srv/ssd_nvm/15hagge/labeled/train_images/",
    train_annotations = "/srv/ssd_nvm/15hagge/labeled/train_segmentation/",
    checkpoints_path = "/tmp/net/",
    epochs = 10
)
