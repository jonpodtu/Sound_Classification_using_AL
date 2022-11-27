from sklearn.model_selection import train_test_split
from active_learning.framework import ALframework
from torch.utils.data import DataLoader
import numpy as np
from datasets import ESC50
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import os
import pandas as pd
from ast import literal_eval


@hydra.main(config_path="config", config_name="WSA1K")
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
        elif cfg.sample_first:
            # If we use the sample first method, we wish to load same indices again (continue run)
            results = pd.read_csv(
                os.path.join(
                    *[
                        f"{n}",
                        cfg.sample_first,
                        "{}_results.csv".format(cfg.initial_budget),
                    ],
                ),
                index_col=0,
            )
            n_results = results[results["train_size"] == cfg.initial_budget]
            initial_indices = literal_eval(n_results["current_indices"].iloc[0])
            print(
                "Loading {} samples with following indices: {}".format(
                    len(initial_indices), initial_indices
                )
            )

        elif cfg.initial_budget < len(all_indices):
            _, initial_indices = train_test_split(
                all_indices,
                test_size=cfg.initial_budget,
                random_state=n,
                shuffle=True,
                stratify=list(audio_labels.iloc[:, 2]),
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

            train_set = ESC50(
                annotations_file=to_absolute_path(cfg.paths.train),
                audio_dir=to_absolute_path(cfg[TaskLearner].train),
                DR=cfg[TaskLearner].DR,
            )

            test_set = ESC50(
                annotations_file=to_absolute_path(cfg.paths.test),
                audio_dir=to_absolute_path(cfg[TaskLearner].test),
                DR=cfg[TaskLearner].DR,
            )
            # Construct datasets
            test_dataloader = DataLoader(
                test_set, batch_size=cfg.batch_size, drop_last=False
            )
            for AL_method in cfg.AL_methods:
                # Path for all output. Whether we want to use the different starts experiment or not
                if cfg.diff_starts:
                    print(
                        "Running diff_start model which means it will lie under different start folders "
                    )
                    diff_starts = f"st_{cfg.initial_budget}"
                    save_path = os.path.join(
                        *[diff_starts, f"{n}", AL_method, TaskLearner]
                    )
                else:
                    save_path = os.path.join(*[f"{n}", AL_method, TaskLearner])
                print("Entering AL loop")
                ActiveLearning.run(
                    train_dataset=train_set,
                    test_dataloader=test_dataloader,
                    TaskLearner=TaskLearner,
                    AL_method=AL_method,
                    save_path=save_path,
                    n=n,
                )
                # Update results
                # results = result_handler(results, new_results)
            # We update at n to avoid information lost at cancelled runs
            # results.to_csv("results.csv")
    print("All iterations are completed!")


if __name__ == "__main__":
    main()
