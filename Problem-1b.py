#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import imageio
import numpy as np
import matplotlib
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm

# learning parameters
batch_size = 512
epochs = 200
sample_size = 64 # fixed sample size for generator
nz = 128 # latent vector size
k = 1 # number of steps to apply to the discriminator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,)),
])
to_pil_image = transforms.ToPILImage()

# Make input, output folders
get_ipython().system('mkdir -p input_orig_objective')
get_ipython().system('mkdir -p outputs_orig_objective')



# Load train data
train_data = datasets.MNIST(
    root='input/data',
    train=True,
    download=True,
    transform=transform
)

if torch.cuda.is_available():
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=32)
else:
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.Linear(self.nz, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )
    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_input = 784
        self.main = nn.Sequential(
            nn.Linear(self.n_input, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = x.view(-1, 784)
        return self.main(x)

generator = Generator(nz).to(device)
discriminator = Discriminator().to(device)
print('##### GENERATOR #####')
print(generator)
print('######################')
print('\n##### DISCRIMINATOR #####')
print(discriminator)
print('######################')

# optimizers
optim_g = optim.Adam(generator.parameters(), lr=0.0002)
optim_d = optim.Adam(discriminator.parameters(), lr=0.0002)

# loss function
criterion = nn.BCELoss() # Binary Cross Entropy loss

losses_g = [] # to store generator loss after each epoch
losses_d = [] # to store discriminator loss after each epoch
images = [] # to store images generatd by the generator

# to create real labels (1s)
def label_real(size):
    data = torch.ones(size, 1)
    return data.to(device)

# to create fake labels (0s)
def label_fake(size):
    data = torch.zeros(size, 1)
    return data.to(device)

# function to create the noise vector
def create_noise(sample_size, nz):
    return torch.randn(sample_size, nz).to(device)

# to save the images generated by the generator
def save_generator_image(image, path):
    save_image(image, path)

# create the noise vector - fixed to track how GAN is trained.
noise = create_noise(sample_size, nz)

torch.manual_seed(7777)

for epoch in range(epochs):
    loss_g = 0.0
    loss_d = 0.0
    for bi, data in tqdm(enumerate(train_loader), total=int(len(train_data) / train_loader.batch_size)):
        X_real = data[0]
        X_real = X_real.to(device)

        for i in range(k):
            optim_d.zero_grad()

            noise_minibatch = create_noise(len(X_real), nz)
            X_fake = generator(noise_minibatch).detach()

            y_real = label_real(len(X_real))
            y_fake = label_fake(len(X_fake))

            y_pred_real = discriminator(X_real)
            y_pred_fake = discriminator(X_fake)

            loss_real = criterion(y_pred_real, y_real)
            loss_fake = criterion(y_pred_fake, y_fake)
            loss_real.backward()
            loss_fake.backward()
            optim_d.step()

            loss_d = loss_d + loss_real + loss_fake

        optim_g.zero_grad()
        noise_minibatch = create_noise(len(X_real), nz)
        X_fake = generator(noise_minibatch)
        y_pred = discriminator(X_fake)

  
        y_fake = label_fake(len(X_fake))
        loss = -1*criterion(y_pred, y_fake)

        loss.backward()
        optim_g.step()
        loss_g = loss

    # create the final fake image for the epoch
    generated_img = generator(noise).cpu().detach()

    # make the images as grid
    generated_img = make_grid(generated_img)

    # save the generated torch tensor models to disk
    save_generator_image(generated_img, f"outputs_orig_objective/gen_img{epoch + 1}.png")
    images.append(generated_img)
    epoch_loss_g = loss_g / bi  # total generator loss for the epoch
    epoch_loss_d = loss_d / bi  # total discriminator loss for the epoch
    losses_g.append(epoch_loss_g.cpu().detach().numpy())
    losses_d.append(epoch_loss_d.cpu().detach().numpy())

    print(f"Epoch {epoch + 1} of {epochs}")
    print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")

    # plot and save the generator and discriminator loss
    plt.figure()
    plt.plot(losses_g, label='Generator loss')
    plt.plot(losses_d, label='Discriminator Loss')
    plt.legend()
    plt.savefig('outputs_orig_objective/loss.png')


print('DONE TRAINING')
torch.save(generator.state_dict(), 'outputs_orig_objective/generator.pth')

# save the generated images as GIF file
imgs = [np.array(to_pil_image(img)) for img in images]
imageio.mimsave('outputs_orig_objective/generator_images.gif', imgs)

# plot and save the generator and discriminator loss
plt.figure()
plt.plot(losses_g, label='Generator loss')
plt.plot(losses_d, label='Discriminator Loss')
plt.legend()
plt.savefig('outputs_orig_objective/loss.png')


# In[ ]:



