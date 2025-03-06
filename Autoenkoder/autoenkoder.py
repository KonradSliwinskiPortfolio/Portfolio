import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import bm3d
from torchsummary import summary

import torch
import torch.cuda
import torch.nn as nn
from torch.nn import Linear, ReLU, MSELoss, L1Loss, Sequential, Conv2d, ConvTranspose2d, MaxPool2d, AdaptiveAvgPool2d, Module, BatchNorm2d, Sigmoid, Dropout
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import random_split
from torchvision import datasets, transforms

import os

gpu_avail = torch.cuda.is_available()
print(f"Is the GPU available? {gpu_avail}")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)

PATH = os.getcwd() + "\\Data\\train"
data_dir = PATH 

transform = transforms.Compose([
    transforms.Resize((255, 255)), 
    transforms.ToTensor()
])
dataset = datasets.ImageFolder(data_dir, transform=transform)
lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]
train_dataset, val_dataset = random_split(dataset, lengths)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)

def imshow(image, ax=None, title=None, normalize=True):
    """Funkcja wyświetlająca obraz zapisany jako tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.cpu().numpy().transpose((1, 2, 0))
    
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        
    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    return ax

images, _ = next(iter(val_dataloader))
imshow(images[1], normalize=False)
plt.show()

noisy_images = (images + torch.normal(0, 0.2, images.shape, device=images.device)).clip(0,1)
imshow(noisy_images[1], normalize=False)
plt.show()

a = 0.7 * torch.ones(images.shape, device=images.device)
bernouilli_noisy_images = images * torch.bernoulli(a)
imshow(bernouilli_noisy_images[1], normalize=False)
plt.show()

a = 5 * torch.ones(images.shape, device=images.device)
p = torch.poisson(a)
p_norm = p / p.max()
poisson_noisy_images = (images + p_norm).clip(0,1)
imshow(poisson_noisy_images[1], normalize=False)
plt.show()

EPS = 1e-8

def PSNR(input, target):
    return -10 * torch.log10(torch.mean((input - target) ** 2, dim=[1, 2, 3]) + EPS)

def MSE(input, target):
    return torch.mean((input - target) ** 2, dim=[1, 2, 3])

print(PSNR(images, noisy_images))
print(PSNR(images, bernouilli_noisy_images))
print(PSNR(images, poisson_noisy_images))

denoised_image = bm3d.bm3d(noisy_images[1].permute(1,2,0).cpu().numpy(), sigma_psd=30/255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
plt.imshow(denoised_image)
plt.show()

bernouilli_denoised_image = bm3d.bm3d(bernouilli_noisy_images[1].permute(1,2,0).cpu().numpy(), sigma_psd=30/255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
plt.imshow(bernouilli_denoised_image)
plt.show()

poisson_denoised_image = bm3d.bm3d(poisson_noisy_images[1].permute(1,2,0).cpu().numpy(), sigma_psd=15/255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING).clip(0,1)
plt.imshow(poisson_denoised_image)
plt.show()

class autoencoders(nn.Module):
    def __init__(self):
        super(autoencoders, self).__init__()
        self.encoder = Sequential(
            Conv2d(3, 32, kernel_size=(3,3), padding="same"),
            ReLU(),
            MaxPool2d((2,2), padding=0),
            Conv2d(32, 64, kernel_size=(3,3), padding="same"),
            ReLU(),
            MaxPool2d((2,2), padding=0),
            Conv2d(64, 128, kernel_size=(3,3), padding="same"),
            ReLU(),
            MaxPool2d((2,2), padding=0)
        )
        self.decoder = Sequential(
            ConvTranspose2d(128, 128, kernel_size=(3,3), stride=2, padding=0),
            ReLU(),
            ConvTranspose2d(128, 64, kernel_size=(3,3), stride=2, padding=0),
            ReLU(),
            ConvTranspose2d(64, 32, kernel_size=(3,3), stride=2, padding=0),
            ReLU(),
            ConvTranspose2d(32, 3, kernel_size=(3,3), stride=1, padding=1),
            Sigmoid()
        )
        
    def forward(self, images):
        x = self.encoder(images)
        x = self.decoder(x)
        return x

model = autoencoders().to(device)
print(summary(model, input_size=(3, 255, 255)))

loss_module = MSELoss()

def eval_model(model, val_dataloader, noise_type, noise_parameter):
    model.eval()
    psnr = []
    mse = []
    with torch.no_grad():
        for images, _ in val_dataloader:
            noisy_images = images  
            if noise_type == "normal":
                noise = torch.normal(0, noise_parameter, images.shape, device=images.device)
                noisy_images = (images + noise).clip(0, 1)
            elif noise_type == "bernoulli":
                a = noise_parameter * torch.ones(images.shape, device=images.device)
                noisy_images = images * torch.bernoulli(a)
            elif noise_type == "poisson":
                a = noise_parameter * torch.ones(images.shape, device=images.device)
                p = torch.poisson(a)
                p_norm = p / p.max()
                noisy_images = (images + p_norm).clip(0, 1)
            noisy_images = noisy_images.to(device)
            images = images.to(device)
            preds = model(images)
            psnr.extend(PSNR(images.cpu().detach(), preds.cpu().detach()))
            mse.extend(MSE(images.cpu().detach(), preds.cpu().detach()))
        print(f"Peak Signal to Noise Ratio:   Mean: {np.array(psnr).mean()} || Std: {np.array(psnr).std()}")
        print(f"Mean Squared Error:   Mean: {np.array(mse).mean()} || Std: {np.array(mse).std()}")
        return np.array(psnr).mean(), np.array(mse).mean()

def train_model(model, noise_type, noise_parameter, optimizer, train_dataloader, val_dataloader, loss_module, target_type="clean", num_epochs=30):
    model.train()
    epoch_num = []
    mse_train = []
    mse_val = []
    psnr_train = []
    psnr_val = []
    for epoch in range(num_epochs):
        for images, _ in train_dataloader:
            targets = torch.clone(images)
            if noise_type == "normal":
                noise = torch.normal(0, noise_parameter, images.shape, device=images.device)
                images = (images + noise).clip(0,1)
            elif noise_type == "bernoulli":
                a = noise_parameter * torch.ones(images.shape, device=images.device)
                images = images * torch.bernoulli(a)
            elif noise_type == "poisson":
                a = noise_parameter * torch.ones(images.shape, device=images.device)
                p = torch.poisson(a)
                p_norm = p / p.max()
                images = (images + p_norm).clip(0,1)
            if target_type == "noisy":
                if noise_type == "normal":
                    noise = torch.normal(0, noise_parameter, targets.shape, device=targets.device)
                    targets = (targets + noise).clip(0,1)
                elif noise_type == "bernoulli":
                    a = noise_parameter * torch.ones(targets.shape, device=targets.device)
                    targets = targets * torch.bernoulli(a)
                elif noise_type == "poisson":
                    a = noise_parameter * torch.ones(targets.shape, device=targets.device)
                    p = torch.poisson(a)
                    p_norm = p / p.max()
                    targets = (targets + p_norm).clip(0,1)
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images)
            loss = loss_module(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 3 == 0:
            print(f"********EPOCH {epoch+1}:********")
            epoch_num.append(epoch+1)
            print("Train set:")
            psnr_val_train, mse_val_train = eval_model(model, train_dataloader, noise_type, noise_parameter)
            psnr_train.append(psnr_val_train)
            mse_train.append(mse_val_train)
            print("Validation set:")
            psnr_val_val, mse_val_val = eval_model(model, val_dataloader, noise_type, noise_parameter)
            psnr_val.append(psnr_val_val)
            mse_val.append(mse_val_val)

normal_ae_mse = autoencoders().to(device)
normal_optimizer_mse = optim.Adam(normal_ae_mse.parameters(), lr=1e-3)
train_model(normal_ae_mse, "normal", 0.2, normal_optimizer_mse, train_dataloader, val_dataloader, loss_module)

normal_ae_mse.eval()
images, _ = next(iter(train_dataloader))
images = images.float().to(device)
output = normal_ae_mse(images)

imshow(images[1].cpu().detach(), normalize=False)

noise = torch.normal(0, 0.2, images.shape, device=images.device)
images = (images + noise).clip(0,1)
                                                          
imshow(images[1].cpu().detach(), normalize=False)

output = normal_ae_mse(images.to(device))
imshow(output[1].cpu().detach(), normalize=False)

images, _ = next(iter(val_dataloader))
images = images.float().to(device)
output = normal_ae_mse(images)
imshow(images[1].cpu().detach(), normalize=False)

noisy_images = (images + torch.normal(0, 0.2, images.shape, device=images.device)).clip(0,1)
imshow(noisy_images[1].cpu().detach(), normalize=False)

output = normal_ae_mse(images)
imshow(output[1].cpu().detach(), normalize=False)

print(MSE(images, output))
print(PSNR(images, output))

a = 5 * torch.ones(images.shape, device=images.device)
p = torch.poisson(a)
p_norm = p / p.max()
images = (images + p_norm).clip(0,1)
imshow(images[1].cpu().detach(), normalize=False)

output = normal_ae_mse(images)
imshow(output[1].cpu().detach(), normalize=False)

normal_noisy_ae = autoencoders().to(device)
normal_noisy_optimizer = optim.Adam(normal_noisy_ae.parameters(), lr=1e-2)
train_model(normal_noisy_ae, "noisy", 0.2, normal_noisy_optimizer, train_dataloader, val_dataloader, loss_module)

images = images.float().to(device)
output = normal_noisy_ae(images)
imshow(images[1].cpu().detach(), normalize=False)
imshow(output[1].cpu().detach(), normalize=False)

images = (images + torch.normal(0, 0.2, images.shape, device=images.device)).clip(0,1)
imshow(images[1].cpu().detach(), normalize=False)
output = normal_noisy_ae(images)
imshow(output[1].cpu().detach(), normalize=False)

normal_noisy_optimizer = optim.Adam(normal_noisy_ae.parameters(), lr=0.005)
loss_module = MSELoss()
train_model(normal_noisy_ae, "clean", 0.2, normal_noisy_optimizer, train_dataloader, val_dataloader, loss_module, num_epochs=10)

imshow(images[1].cpu().detach(), normalize=False)
noisy_images = (images + torch.normal(0, 0.2, images.shape, device=images.device)).clip(0,1)
imshow(noisy_images[1].cpu().detach(), normalize=False)
output = normal_noisy_ae(noisy_images)
imshow(output[1].cpu().detach(), normalize=False)

# Zapisanie wag pierwszego modelu
state_dict = normal_ae_mse.state_dict()
print(state_dict)
file_path = "normal_ae_mse_30epochs.tar"
torch.save(state_dict, file_path)

loss_module = L1Loss()
normal_ae_mae = autoencoders().to(device)
normal_optimizer_mae = optim.Adam(normal_ae_mae.parameters(), lr=1e-3)
train_model(normal_ae_mae, "normal", 0.2, normal_optimizer_mae, train_dataloader, val_dataloader, loss_module)

images = images.float().to(device)
imshow(images[1].cpu().detach(), normalize=False)
output = normal_ae_mae(images)
imshow(output[1].cpu().detach(), normalize=False)

noisy_images = (images.cpu().detach() + torch.normal(0, 0.2, images.shape)).clip(0,1).to(device)
imshow(noisy_images[1].cpu().detach(), normalize=False)
output = normal_ae_mae(noisy_images)
imshow(output[1].cpu().detach(), normalize=False)

state_dict = torch.load("normal_ae_mse_30epochs.tar")
normal_ae_mse = autoencoders().to(device)
normal_ae_mse.load_state_dict(state_dict)
output1 = normal_ae_mse(noisy_images)
imshow(output1[1].cpu().detach(), normalize=False)

state_dict = normal_ae_mae.state_dict()
normal_ae_mse = autoencoders().to(device)
state_dict = torch.load("normal_ae_mse_30epochs.tar")
normal_ae_mse.load_state_dict(state_dict)

psnr = []
with torch.no_grad():
    for i, (images, _) in enumerate(val_dataloader):
        noise = torch.normal(0, 0.2, images.shape, device=images.device)
        noisy_images = (images + noise).clip(0,1).to(device)
        images = images.to(device)
        preds = normal_ae_mse(noisy_images)
        if i < 4:
            print(PSNR(images.cpu().detach(), preds.cpu().detach()))
        psnr.extend(PSNR(images.cpu().detach(), preds.cpu().detach()))
mse_psnr = np.array(psnr)
print(f"The mean of the PSNR is {mse_psnr.mean()} and the standard deviation of the PSNR is {mse_psnr.std()}")

normal_ae_mae = autoencoders().to(device)
state_dict = torch.load("normal_ae_mse_30epochs.tar")
normal_ae_mae.load_state_dict(state_dict)
psnr = []
with torch.no_grad():
    for i, (images, _) in enumerate(val_dataloader):
        noise = torch.normal(0, 0.2, images.shape, device=images.device)
        noisy_images = (images + noise).clip(0,1).to(device)
        images = images.to(device)
        preds = normal_ae_mae(noisy_images)
        if i < 4:
            print(PSNR(images.cpu().detach(), preds.cpu().detach()))
        psnr.extend(PSNR(images.cpu().detach(), preds.cpu().detach()))
mae_psnr = np.array(psnr)
print(f"The mean of the PSNR is {mae_psnr.mean()} and the standard deviation of the PSNR is {mae_psnr.std()}")

psnr = []
with torch.no_grad():
    for i, (images, _) in enumerate(val_dataloader):
        noise = torch.normal(0, 0.2, images.shape, device=images.device)
        noisy_images = (images + noise).clip(0,1)
        for j in range(images.shape[0]):
            img_noisy = noisy_images[j,:,:,:].permute(1,2,0).cpu().numpy()
            bm3d_denoised = bm3d.bm3d(img_noisy, sigma_psd=30/255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
            bm3d_denoised = torch.tensor(bm3d_denoised).permute(2,0,1).unsqueeze(0)
            psnr.append(PSNR(images[j,:,:,:].unsqueeze(0), bm3d_denoised))
        if i < 4:
            print(psnr)
bm3d_psnr = torch.tensor(psnr)
print(f"The mean of the PSNR is {bm3d_psnr.mean()} and the standard deviation of the PSNR is {bm3d_psnr.std()}")
