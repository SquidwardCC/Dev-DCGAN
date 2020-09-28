import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
from model_utils import Discriminator, Generator
import os

if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")

iterative_checkpoint = True
lr = 0.0001
batch_size = 200
image_size = 128
channels_img = 3
channels_noise = 256
num_epochs = 1000
features_d = 32
features_g = 32

my_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    ])

def save(iteration):
    iter_dir = "checkpoints/"+str(iteration)
    os.makedirs(iter_dir, exist_ok=True)
    torch.save(netD.state_dict(), f'{iter_dir}/netD')
    torch.save(netG.state_dict(), f'{iter_dir}/netG')
    torch.save(optimizerD.state_dict(), f'{iter_dir}/optD')
    torch.save(optimizerG.state_dict(), f'{iter_dir}/optG')

dataset = datasets.ImageFolder(root='datain', transform=my_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda:", torch.cuda.is_available())

netD = Discriminator(channels_img, features_d).to(device)
netG = Generator(channels_noise, channels_img, features_g).to(device)


optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

netG.train()
netD.train()

criterion = nn.BCELoss()

real_label = 1
fake_label = 0

fixed_noise = torch.randn(64, channels_noise, 1, 1).to(device)
writer_real = SummaryWriter(f'runs/training/test_real')
writer_fake = SummaryWriter(f'runs/training/test_fake')

def train(resume=False, init_epoch=0):
    print("Training")

    if resume == True:
        print("Loading from saved models...")
        netD.load_state_dict(torch.load(f'checkpoints/netD'))
        netG.load_state_dict(torch.load(f'checkpoints/netG'))
        optimizerD.load_state_dict(torch.load(f'checkpoints/optD'))
        optimizerG.load_state_dict(torch.load(f'checkpoints/optG'))



    for epoch in range(num_epochs-init_epoch):
        for batch_idx, (data, targets) in enumerate(dataloader):
            data = data.to(device)
            batch_size = data.shape[0]

            # Train discriminator
            netD.zero_grad()
            label = (torch.ones(batch_size)*0.9).to(device)
            output = netD(data).reshape(-1)
            lossD_real = criterion(output, label)
            D_x = output.mean().item()

            noise = torch.randn(batch_size, channels_noise, 1, 1,).to(device)
            fake = netG(noise)
            label = (torch.ones(batch_size)*0.1).to(device)

            output = netD(fake.detach()).reshape(-1)
            lossD_fake = criterion(output, label)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

            # Train generator
            netG.zero_grad()
            label = torch.ones(batch_size).to(device)
            output = netD(fake).reshape(-1)
            lossG = criterion(output, label)
            lossG.backward()
            optimizerG.step()

            if batch_idx % 5 == 0:
                print(f'Epoch [{epoch+init_epoch}/{num_epochs-init_epoch}] Batch {batch_idx}/{len(dataloader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f} D(x): {D_x:.4f}')

            with torch.no_grad():
                fake = netG(fixed_noise)

                img_grid_real = torchvision.utils.make_grid(data[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                writer_real.add_image('Real Images', img_grid_real)
                writer_real.add_image('Fake Images', img_grid_fake)
                
        if iterative_checkpoint:
            save(epoch)
        else:
            save("recent")

train(resume=False, init_epoch=0)