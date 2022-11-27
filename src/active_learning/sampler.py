import torch
import numpy as np
from active_learning.vaal.model import VAE, Discriminator
from active_learning.vaal.solver import Solver
from datasets import ESC50
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader
import os


def sample_uncertainty(preds: torch.Tensor, indices: list, n_samples: int = 1,) -> list:
    """Sample given predictions and indices. set n_sample to 0 if only preds should be returned 

    PARAMETERS
    ----------
    preds : Torch tensor with all predictions
    indices: list of all indices in the same order as they are arranged in preds
    n_samples: number of samples to query add a time

    """

    # We define a small epsilon value which we add to our preds to avoid log(0) = -inf
    eps = torch.finfo(torch.float32).eps
    entropy = -torch.sum(preds * torch.log2(preds + eps), axis=1)
    # If we only want we predictions, we simply don't need the samples
    if n_samples == 0:
        return entropy
    _, querry_indices = torch.topk(entropy, n_samples)
    querry_pool_indices = np.asarray(indices)[querry_indices]
    querry_pool_indices = querry_pool_indices.tolist()

    if n_samples == 1:
        return [querry_pool_indices]

    return entropy, querry_pool_indices


def acquistion_max_entropy(entropy, indices):
    # If we only want we predictions, we simply don't need the samples
    _, querry_indices = torch.topk(entropy, 1)
    choice = indices[querry_indices]

    return int(choice)


def acquistion_random(indices):
    return int(np.random.choice(indices, size=1, replace=False))


def acquisition_mixed_strategy(indices, entropy, p=0.5):
    z = np.random.binomial(1, p=p)
    if z == 1:
        return acquistion_max_entropy(entropy, indices)
    else:
        return acquistion_random(indices)


def sample_uncertainRandom(
    preds: torch.Tensor, indices: list, n_samples: int = 1, rand_prob: float = 0.5
) -> list:
    """Sample given predictions and indices. set n_sample to 0 if only preds should be returned 

    PARAMETERS
    ----------
    preds : Torch tensor with all predictions
    indices: list of all indices in the same order as they are arranged in preds
    n_samples: number of samples to query add a time

    """
    eps = torch.finfo(torch.float32).eps
    entropy = -torch.sum(preds * torch.log2(preds + eps), axis=1)

    querry_pool_indices = np.empty(n_samples, dtype=int)
    indices = np.asarray(indices)
    for i in range(n_samples):
        choice = acquisition_mixed_strategy(indices, entropy, p=rand_prob)
        querry_pool_indices[i] = choice
        idx_to_keep = np.where(indices != choice)[0]

        indices = indices[idx_to_keep]
        entropy = entropy[idx_to_keep]

    return entropy, querry_pool_indices.tolist()


def sample_with_vaal(
    model_save, device, cfg, query_dataloader, pool_dataloader, n_samples: int = 1
) -> list:
    """Sample given predictions and indices. set n_sample to 0 if only preds should be returned 

    PARAMETERS
    ----------
    preds : Torch tensor with all predictions
    indices: list of all indices in the same order as they are arranged in preds
    n_samples: number of samples to query add a time

    """
    train_set = ESC50(
        annotations_file=to_absolute_path(cfg.paths.train),
        audio_dir=to_absolute_path(cfg.VAAL.train),
        DR=False,
    )

    query_dataloader = DataLoader(
        train_set,
        sampler=query_dataloader.sampler,
        batch_size=cfg.batch_size,
        drop_last=False,
    )
    pool_dataloader = DataLoader(
        train_set,
        sampler=pool_dataloader.sampler,
        batch_size=cfg.batch_size,
        drop_last=False,
    )

    solver = Solver(cfg=cfg)
    vae = VAE(32)
    discriminator = Discriminator(32)
    vae, discriminator, loss_df = solver.train(
        device, query_dataloader, vae, discriminator, pool_dataloader
    )

    # SAVE MODEL
    vae.eval()
    discriminator.eval()
    if model_save:
        if not os.path.exists(model_save):
            os.makedirs(model_save)
        torch.save(
            vae.state_dict(), os.path.join(model_save, "vae.pt"),
        )
        torch.save(
            discriminator.state_dict(), os.path.join(model_save, "discriminator.pt"),
        )

    train_indices, train_preds, train_mu = solver.sample_for_labeling(
        vae, discriminator, query_dataloader, 0, device
    )
    pool_indices, pool_preds, querry_indices, pool_mu = solver.sample_for_labeling(
        vae, discriminator, pool_dataloader, cfg.n_samples, device
    )

    if n_samples == 1:
        querry_indices = [querry_indices]
    else:
        querry_indices = list(querry_indices)

    # train_indices, train_mu, train_preds = solver.sample_for_labeling(
    #    vae, discriminator, query_dataloader, 0, device
    # )

    return (
        (pool_indices, pool_mu, pool_preds),
        (
            train_indices,
            train_mu,
            train_preds,
        ),  # (train_indices, train_mu, train_preds)
        pool_preds,
        querry_indices,
        discriminator,
        vae,
        loss_df,
    )
