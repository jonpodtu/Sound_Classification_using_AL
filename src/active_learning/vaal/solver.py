import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tqdm import tqdm

# All code taken from the VAAL github https://github.com/sinhasam/vaal
class Solver:
    def __init__(self, cfg):
        # self.test_dataloader = test_dataloader
        self.budget = cfg.budget
        self.dataset_size = cfg.VAAL.dataset_size
        self.batch_size = cfg.batch_size

        # Hardcoded training parameters for VAE and Discriminator
        self.num_vae_steps = 2
        self.num_adv_steps = 1
        self.beta = 1
        self.adversary_param = 1
        self.train_epochs = cfg.VAAL.epochs

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, _ in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader:
                    yield img

    def train(
        self, device, querry_dataloader, vae, discriminator, unlabeled_dataloader
    ):
        self.train_iterations = (
            self.dataset_size * self.train_epochs
        ) // self.batch_size

        labeled_data = self.read_data(querry_dataloader)
        unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)

        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)

        vae.train()
        discriminator.train()

        vae = vae.to(device)
        discriminator = discriminator.to(device)

        i = 0
        losses = np.empty(4)

        print("Training VAAL...")
        for iter_count in tqdm(range(self.train_iterations)):

            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)

            labeled_imgs = labeled_imgs.to(device)
            unlabeled_imgs = unlabeled_imgs.to(device)
            labels = labels.to(device)

            # VAE step
            for count in range(self.num_vae_steps):
                recon, z, mu, logvar = vae(labeled_imgs)
                unsup_loss = self.vae_loss(labeled_imgs, recon, mu, logvar, self.beta)
                unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
                transductive_loss = self.vae_loss(
                    unlabeled_imgs, unlab_recon, unlab_mu, unlab_logvar, self.beta
                )

                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)

                lab_real_preds = torch.ones(labeled_imgs.size(0))
                unlab_real_preds = torch.ones(unlabeled_imgs.size(0))

                lab_real_preds = lab_real_preds.to(device)
                unlab_real_preds = unlab_real_preds.to(device)

                dsc_loss = self.bce_loss(
                    labeled_preds,
                    torch.reshape(lab_real_preds, (lab_real_preds.shape[0], 1)),
                ) + self.bce_loss(
                    unlabeled_preds,
                    torch.reshape(unlab_real_preds, (unlab_real_preds.shape[0], 1)),
                )
                total_vae_loss = (
                    unsup_loss + transductive_loss + self.adversary_param * dsc_loss
                )
                # print(f"Total VAE loss: {total_vae_loss}")
                optim_vae.zero_grad()
                total_vae_loss.backward()
                optim_vae.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.num_vae_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    labeled_imgs = labeled_imgs.to(device)
                    unlabeled_imgs = unlabeled_imgs.to(device)
                    labels = labels.to(device)

            # Discriminator step
            for count in range(self.num_adv_steps):
                with torch.no_grad():
                    _, _, mu, _ = vae(labeled_imgs)
                    _, _, unlab_mu, _ = vae(unlabeled_imgs)

                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)

                lab_real_preds = torch.ones(labeled_imgs.size(0))
                unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

                lab_real_preds = lab_real_preds.to(device)
                unlab_fake_preds = unlab_fake_preds.to(device)

                dsc_loss = self.bce_loss(
                    labeled_preds,
                    torch.reshape(lab_real_preds, (lab_real_preds.shape[0], 1)),
                ) + self.bce_loss(
                    unlabeled_preds,
                    torch.reshape(unlab_fake_preds, (unlab_fake_preds.shape[0], 1)),
                )
                # print(f"Total DSC loss: {dsc_loss}")
                optim_discriminator.zero_grad()
                dsc_loss.backward()
                optim_discriminator.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.num_adv_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    labeled_imgs = labeled_imgs.to(device)
                    unlabeled_imgs = unlabeled_imgs.to(device)
                    labels = labels.to(device)

            if iter_count % 100 == 0:
                print(f"Iteration {iter_count}/{self.train_iterations}")
                print("Current vae model loss: {:.4f}".format(total_vae_loss.item()))
                print(
                    "Current discriminator model loss: {:.4f}".format(dsc_loss.item())
                )

            if i == 0:
                losses = np.array(
                    [
                        unsup_loss.item(),
                        transductive_loss.item(),
                        total_vae_loss.item(),
                        dsc_loss.item(),
                    ]
                )
            else:
                losses = np.vstack(
                    [
                        losses,
                        [
                            unsup_loss.item(),
                            transductive_loss.item(),
                            total_vae_loss.item(),
                            dsc_loss.item(),
                        ],
                    ]
                )

            i += 1

        loss_df = pd.DataFrame(
            losses,
            columns=[
                "Unsupervised Loss",
                "Transductive Loss",
                "Total VAE Loss",
                "Discriminator Loss",
            ],
        )
        return vae, discriminator, loss_df

    def sample_for_labeling(
        self, vae, discriminator, unlabeled_dataloader, n_samples, device
    ):
        # From the Adversary Sampler function

        all_preds = []
        all_indices = []
        all_mu = []

        for images, _, indices in unlabeled_dataloader:
            images = images.to(device)

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)
            all_mu.extend(mu)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        if n_samples > 0:
            _, querry_indices = torch.topk(all_preds, n_samples)
            querry_pool_indices = np.asarray(all_indices)[querry_indices]

            return all_indices, all_preds, querry_pool_indices, all_mu

        else:
            return all_indices, all_preds, all_mu

    def validate(self, task_model, loader):
        # Not used
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels, _ in loader:
            if self.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100

    def test(self, task_model):
        # Not used
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels in self.test_dataloader:
            if self.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100

    def vae_loss(self, x, recon, mu, logvar, beta):
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD
