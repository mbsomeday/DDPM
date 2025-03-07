# https://github.com/dome272/Diffusion-Models-pytorch

import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
import logging
from torch.utils.tensorboard import SummaryWriter

from MY_DICT import DICT
from datasets import pedCls_Dataset

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        # logging.info(f"Sampling {n} new images....")
        print(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device

    # dataloader = get_data(args)
    ped_ds = pedCls_Dataset(dict=DICT, ds_name_list=['D4'], txt_name='augmentation_train.txt', img_size=args.image_size, get_num=-1)
    dataloader = DataLoader(ped_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 若是从头训练
    if not args.reload:
        model = UNet().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        start_epoch = 0

    # 若是恢复训练
    else:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
        model = UNet().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.1)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint['epoch']


    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(start_epoch, args.epochs):
        # logging.info(f"Starting epoch {epoch}:")
        print(f"Starting epoch {epoch}:")
        epoch_loss = 0

        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

            epoch_loss += loss.item()

        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        # torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

        # 模型每个epoch完整保存
        save_model_name = "ep%03d-MSE%.6f.pth"% (epoch + 1, epoch_loss / len(dataloader))
        save_model_path = os.path.join("models", args.run_name, save_model_name)
        state = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch
                 }
        torch.save(state, save_model_path)



def launch():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path')
    parser.add_argument('--reload')


    args = parser.parse_args()

    args.run_name = "DDPM_Uncondtional"
    args.epochs = 100
    args.batch_size = 16

    args.image_size = 224


    # args.image_size = 128

    # args.dataset_path = r"C:\Users\dome\datasets\landscape_img_folder"

    args.device = "cuda"
    args.lr = 3e-4



    train(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet().to(device)
    # ckpt = torch.load("./working/orig/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # x = diffusion.sample(model, 8)
    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()
