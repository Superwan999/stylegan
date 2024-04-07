import os.path

import numpy as np
import torch
import torch.nn as nn
from network import StyleGenerator, StyleDiscriminator
import torch.optim as optim
from loss import r1_penalty, gradient_penalty_loss
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from process_data import CustomDataset
import gc


class Trainer:
    def __init__(self, opts):
        self.opts = opts

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # self.G = StyleGenerator()
        # self.D = StyleDiscriminator()
        self.G = StyleGenerator(resolution=opts['img_size'],
                                fmap_base=opts['fmap_base'],
                                mapping_fmaps=opts['latent_size'],
                                fmap_max=opts['latent_size'])
        self.D = StyleDiscriminator(resolution=opts['img_size'], fmap_base=opts['fmap_base'], fmap_max=opts['latent_size'])

        reume_path = opts.get('resume', None)
        if reume_path is not None and os.path.exists(opts['resume']):
            print(f"Load pretrained weight !")
            state = torch.load(opts['resume'])
            self.G.load_state_dict(state['G'])
            self.D.load_state_dict(state['D'])

        self.D = self.D.to(self.device)
        self.G = self.G.to(self.device)

        self.data_loader = DataLoader(dataset=CustomDataset(opts['data_root'], opts['img_size']),
                                      batch_size=opts['batch_size'],
                                      shuffle=True)

        self.optimizer_G = optim.Adam(self.G.parameters(), lr=opts['lr_G'], betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=opts['lr_D'], betas=(0.5, 0.999))

        self.scheduler_G = optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.99)
        self.scheduler_D = optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.99)

        self.fix_z = torch.randn([opts['batch_size'], opts['latent_size']]).to(self.device)

        self.soft_plus = nn.Softplus()

        self.epoch_loss_D_list = [ ]
        self.epoch_loss_G_list = [ ]

    def runAndOptimize(self, epoch):
        bar = tqdm(self.data_loader)
        loss_D_list = []
        loss_G_list = []
        for i, real_img in enumerate(bar):
            real_img = real_img.to(self.device)

            for p in self.D.parameters():
                p.requires_grad = True

            self.D.zero_grad()
            real_logit = self.D(real_img)

            z = torch.randn([real_img.size(0), self.opts['latent_size']]).to(self.device)
            fake_img = self.G(z)
            fake_logit = self.D(fake_img.detach())

            d_loss = self.soft_plus(fake_logit).mean() + self.soft_plus(-real_logit).mean()

            if self.opts['r1_gamma'] != 0.0:
                real_img.requires_grad = True
                real_pred = self.D(real_img)
                r1_penalty_loss = r1_penalty(real_pred=real_pred, real_img=real_img)
                d_loss += r1_penalty_loss * (self.opts['r1_gamma'] * 0.5)

            loss_D_list.append(d_loss.item())

            d_loss.backward()
            self.optimizer_D.step()


            # Update Generator
            if i % self.opts['opt_g_freq'] == 0:
                for p in self.D.parameters():
                    p.requires_grad = False

                self.G.zero_grad()
                fake_logit = self.D(fake_img)
                g_loss = self.soft_plus(-fake_logit).mean()
                loss_G_list.append(g_loss.item())

                g_loss.backward()
                self.optimizer_G.step()

            if i % self.opts['print_freq'] == 0:
                print(f"===> Epoch: {epoch}: [{i + 1}/{len(self.data_loader)}] [G_loss: {loss_G_list[-1]}], [D_loss: {loss_D_list[-1]}]")

            if i % self.opts['save_img_freq'] == 0:
                with torch.no_grad():
                    fake_img = self.G(self.fix_z).detach().cpu()
                    self.saveImage(fake_img, i, epoch)


        self.epoch_loss_D_list.append(sum(loss_D_list) / len(loss_D_list))
        self.epoch_loss_G_list.append(sum(loss_G_list) / len(loss_G_list))


    def saveImage(self, fake_img, iters, epoch):

        save_file = os.path.join(self.opts['save_img_path'], f'{epoch}_{iters}')
        os.makedirs(save_file, exist_ok=True)
        for i in range(fake_img.size(0)):
            img = fake_img[i].permute(1, 2, 0).numpy()
            img = np.clip(img, 0.0, 1.0) * 255
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            img.save(fp=os.path.join(save_file, f"{i}.png"))

    def plotLossCurve(self):
        plt.figure()
        plt.plot(self.epoch_loss_D_list, '-')
        plt.title("Loss Curve Discriminator")
        save_loss_fig_file = os.path.join(self.opts['save_path'], 'loss_fig')
        os.makedirs(save_loss_fig_file, exist_ok=True)
        plt.savefig(os.path.join(save_loss_fig_file, 'loss_curve_discriminator.png'))

        plt.figure()
        plt.plot(self.epoch_loss_G_list, 'o')
        plt.title("Loss Curve Generator")
        plt.savefig(os.path.join(save_loss_fig_file, 'loss_curve_generator.png'))

    def train(self):
        for epoch in range(1, self.opts['num_epochs'] + 1):
            self.runAndOptimize(epoch)
            gc.collect()

            state = {
                'G': self.G.state_dict(),
                'D': self.D.state_dict(),
            }
            save_weights_file = os.path.join(self.opts['save_path'], 'weights')
            os.makedirs(save_weights_file, exist_ok=True)
            torch.save(state, os.path.join(save_weights_file, f"epoch_{epoch}.pth"))

            self.scheduler_D.step()
            self.scheduler_G.step()

        self.plotLossCurve()

if __name__ == "__main__":
    import argparse
    import yaml

    gc.collect()

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', default='./stylegan.yml', type=str, help="option path for training")
    args = parser.parse_args()

    with open(args.opt) as f:
        opts = yaml.load(f, Loader=yaml.SafeLoader)
    trainer = Trainer(opts)
    trainer.train()
