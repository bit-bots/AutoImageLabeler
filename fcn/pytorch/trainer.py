import os
import torchvision
import torch
from torch.utils.data import DataLoader
from vgg16WithUpsample import vgg16
from dataset import roboCupDatasets, Rescale, ToTensor, Normalize

# create directory to save weights in
# TODO
new_folder = False
while not new_folder:
    save_folder = input("Name the folder you want the weights to be saved in."
                        " Will be saved relative to current position, in a folder called weights.\n")
    save_folder = "weights/" + save_folder
    if not os.path.exists(f"./{save_folder}"):
        os.makedirs(f"./{save_folder}")
        new_folder = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"We are training on {device}")

load_path = "vgg16-conv.pth"
if load_path:
    checkpoint = torch.load(load_path)
    # TODO comment in when we want to load a from this trainer saved net
    # model_dict = checkpoint['model_state_dict']
    model = vgg16(checkpoint)
    print("We loaded a checkpoint")
else:
    model = vgg16()

model.to(device)



train_dataset = roboCupDatasets(transform=torchvision.transforms.Compose([Rescale(), ToTensor(), Normalize()]))
train_dl = DataLoader(train_dataset, batch_size=2, shuffle=True)

epochs = 35

opt = torch.optim.Adam(model.parameters(), lr=0.0001)

with open(f"./{save_folder}/stats.csv", "w+") as f:
    f.write("train_loss\n")

for epoch in range(epochs):
    model.train()
    train_loss = []
    for i, sample in enumerate(train_dl):
        with torch.set_grad_enabled(True):
            opt.zero_grad()
        img, label = sample
        img, label = img.to(device), label.to(device)
        pred = model(img)

        weight = 1.0 / ((1.1 - label) ** 2.0)
        print('pred:')
        print(pred.shape)
        print('label:')
        print(label.shape)
        loss = (weight * torch.nn.functional.mse_loss(pred, label, reduction='none')).mean()
        train_loss.append(loss)

        loss.backward()
        opt.step()
        print(f"epoch {epoch} batch {i + 1}/{len(train_dl)} | loss = {loss} \r", end="")
    model.eval()
    save_path = f"./{save_folder}/my_model_epoch{epoch}.pt"
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss}, save_path)

    with open(f"./{save_folder}/stats.csv", "a") as f:
        f.write(f"{train_loss}\n")
