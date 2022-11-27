from sklearn.model_selection import train_test_split
from active_learning.framework import ALframework
from torch.utils.data import DataLoader
import numpy as np
from datasets import ESC50, Iris
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from utils.helper import result_handler
import os
import pandas as pd


@hydra.main(config_path="config", config_name="Iris_config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print("Everything will be dropped in the working directory: {}".format(os.getcwd()))

    results = None

    if isinstance(cfg.n_avg, int):
        n_avg = range(cfg.n_avg)
        # If we give a list, we just want to go through the ns given in list
    else:
        n_avg = cfg.n_avg

    # MAIN LOOP
    for n in tqdm(n_avg):
        # We sample pur initial indicies according to our initial budget.
        # We want to use the same initial indicies for all methods for fairness
        audio_labels = pd.read_csv(to_absolute_path(cfg.paths.train))

        all_indices = np.arange(len(audio_labels))
        if not isinstance(cfg.initial_budget, int):
            initial_indices = list(cfg.initial_budget)
            cfg.initial_budget = len(initial_indices)
            print(
                f"Using predifined list of indices as initial indices. Initial budget is {cfg.initial_budget} and the indices are \n{initial_indices}"
            )
        elif cfg.initial_budget < len(all_indices):
            _, initial_indices = train_test_split(
                all_indices,
                test_size=cfg.initial_budget,
                random_state=n,
                shuffle=True,
                stratify=list(audio_labels.iloc[:, 4]),
            )
        else:
            initial_indices = all_indices

        all_indices = set(all_indices)

        ActiveLearning = ALframework(
            cfg=cfg, initial_indices=initial_indices, all_indices=all_indices,
        )
        print("Entering main loop")
        for TaskLearner in cfg.TaskLearners:
            if cfg.use_navg_as_DR:
                cfg[TaskLearner].DR = n

            train_set = Iris(data=to_absolute_path(cfg.paths.train), features=[1, 3])

            test_set = Iris(data=to_absolute_path(cfg.paths.test), features=[1, 3])
            # Construct datasets
            test_dataloader = DataLoader(
                test_set, batch_size=cfg.batch_size, drop_last=False
            )

            for AL_method in cfg.AL_methods:
                # Path for all output
                save_path = os.path.join(*[f"{n}_run", AL_method, TaskLearner])
                print("Entering AL loop")
                new_results = ActiveLearning.run(
                    train_dataset=train_set,
                    test_dataloader=test_dataloader,
                    TaskLearner=TaskLearner,
                    AL_method=AL_method,
                    save_path=save_path,
                    n=n,
                )
                # Update results
                results = result_handler(results, new_results)
            # We update at n to avoid information lost at cancelled runs
            results.to_csv("results.csv")
    print("All iterations are completed!")


if __name__ == "__main__":
    main()
