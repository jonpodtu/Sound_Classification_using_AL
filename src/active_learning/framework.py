import numpy as np
import torch
from utils.model_tools import train, inference, initialize_model
import pandas as pd
from torch.utils.data import DataLoader, SubsetRandomSampler
from utils.plot_functions import plot_simple, plot_vaal, plot_in_pc
from ast import literal_eval
from active_learning.sampler import (
    sample_uncertainty,
    sample_with_vaal,
    sample_uncertainRandom,
)
import os
from hydra.utils import to_absolute_path


class ALframework:
    def __init__(self, cfg, initial_indices, all_indices,) -> None:
        try:
            self.n_iterations = int(
                ((cfg.budget - len(initial_indices)) / cfg.n_samples) + 1
            )
        except ZeroDivisionError:
            self.n_iterations = 1

        self.initial_indices = initial_indices
        self.all_indices = all_indices
        self.cfg = cfg

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))

    def run(
        self,
        train_dataset,
        test_dataloader,
        n: int,
        TaskLearner: str = "Simple",
        AL_method: str = "Random",
        save_path: str = "",
    ):
        """Run n_iterations of active learning or random sampling.
        
        PARAMETERS
        ----------
        train_dataset : a torch dataset consisting of all elements in the train data 
        query_dataloader: query dataloader knows which samples from train_dataset to sample from and which are still in the pool
        test_dataloader: dataloader form the static test set
        TaskLearner : type string, either "Linear", "CNN" or "AST"
        AL_method : type string, either "Random", "Uncertainty" or "Advanced"

        RETURNS
        ----------
        results : pandas array    
        """

        current_indices = self.initial_indices
        results = None

        # Define Query dataloader
        query_sampler = SubsetRandomSampler(current_indices)
        query_dataloader = DataLoader(
            train_dataset,
            sampler=query_sampler,
            batch_size=self.cfg.batch_size,
            drop_last=True,
        )

        for i in range(self.n_iterations):
            model_save = os.path.join(save_path, "models")
            # LETS MAKE A POOL
            pool_indices = np.setdiff1d(list(self.all_indices), current_indices)
            pool_sampler = SubsetRandomSampler(pool_indices)
            pool_dataloader = DataLoader(
                train_dataset,
                sampler=pool_sampler,
                batch_size=self.cfg.batch_size,
                drop_last=False,
            )
            print(
                "Iteration: {} \t Trainsize: {} \t Poolsize: {} \t Budget: {}".format(
                    i, len(current_indices), len(pool_indices), self.cfg.budget
                )
            )

            # We can predefine a model through sample_first, meaning that we will pass the first training.
            if self.cfg.sample_first and i == 0:
                model, _ = initialize_model(
                    model_type=TaskLearner,
                    n_features=self.cfg[TaskLearner].DR,
                    n_classes=self.cfg["n_class"],
                    seed=n,
                )
                model_path = os.path.join(
                    *[
                        f"{n}",
                        self.cfg.sample_first,
                        "models",
                        "{}_model.pt".format(len(current_indices)),
                    ]
                )
                model.load_state_dict(torch.load(model_path))
                model.eval()
                print("Loaded model from: ", to_absolute_path(model_path))

                # We want a copy of the results
                results = pd.read_csv(
                    os.path.join(
                        *[
                            f"{n}",
                            self.cfg.sample_first,
                            "{}_results.csv".format(len(current_indices)),
                        ],
                    ),
                    index_col=0,
                )

            else:
                # 1. Fit the model
                model, new_results = train(
                    TaskLearner,
                    train_dataset,
                    current_indices,
                    device=self.device,
                    test_dataloader=test_dataloader,
                    cfg=self.cfg,
                    seed=n,
                )

                # Update results
                results = self.result_handler(
                    results,
                    new_results,
                    TaskLearner,
                    AL_method,
                    len(current_indices),
                    len(pool_indices),
                    n=n,
                )

            if not os.path.exists(model_save):
                os.makedirs(model_save)
            results.to_csv(
                os.path.join(save_path, "{}_results.csv".format(len(current_indices)))
            )
            torch.save(
                model.state_dict(),
                os.path.join(
                    model_save, "{}_model.pt".format(str(len(current_indices)))
                ),
            )

            # 3. Select queries
            if AL_method == "Random":
                # Random sampling without replacement
                # In this specific case, our indices are in the same permutation as original
                indices = pool_indices
                queries = (
                    np.random.choice(indices, size=self.cfg.n_samples, replace=False)
                ).tolist()

            elif self.cfg.load_queries:
                query_dataframe = pd.read_csv(to_absolute_path(self.cfg.load_queries))
                queries = query_dataframe.Queries[query_dataframe.Iteration == i].apply(
                    literal_eval
                )[0]

            elif AL_method == "Uncertainty":
                predictions, indices = inference(model, self.device, pool_dataloader)
                pool_preds, queries = sample_uncertainty(
                    preds=predictions, indices=indices, n_samples=self.cfg.n_samples
                )
                discriminator = None
                vae = None
            elif AL_method == "Uncertain+Random":
                predictions, indices = inference(model, self.device, pool_dataloader)
                pool_preds, queries = sample_uncertainRandom(
                    preds=predictions, indices=indices, n_samples=self.cfg.n_samples
                )
            elif AL_method == "VAAL":
                (
                    pool_info,
                    train_info,
                    pool_preds,
                    queries,
                    discriminator,
                    vae,
                    loss_df,
                ) = sample_with_vaal(
                    False,
                    self.device,
                    n_samples=self.cfg.n_samples,
                    cfg=self.cfg,
                    query_dataloader=query_dataloader,
                    pool_dataloader=pool_dataloader,
                )

                # Save loss_df as csv file:
                loss_df.to_csv(
                    os.path.join(
                        save_path, "{}_vaal_loss.csv".format(len(current_indices))
                    )
                )

            else:
                raise ValueError(
                    "The chosen AL_method has to be either 'Random', 'Uncertainty' or 'VAAL'"
                )

            # produce uncertainty plot:
            if self.cfg.use_plots and (len(queries) > 0):
                """
                plot_in_pc(
                    self.device,
                    AL_method,
                    pool_indices,
                    current_indices,
                    queries,
                    i,
                    save_path,
                    train_dataset,
                    model=model,
                    vae=vae,
                    discriminator=discriminator,
                )
                """
                if AL_method == "VAAL":
                    plot_vaal(
                        discriminator,
                        pool_info,
                        train_info,
                        queries,
                        i,
                        save_path,
                        train_dataset,
                    )

                elif TaskLearner == "Simple":
                    plot_simple(
                        self.device,
                        self.cfg,
                        model,
                        current_indices,
                        pool_indices,
                        queries,
                        indices,
                        save_path,
                        pool_preds,
                        i,
                    )

                else:
                    print(
                        "No plotting was done as neither VAAL or simple model was used - or you didn't sample any points!"
                    )

            if i < self.n_iterations:
                # 4. Update trainset
                current_indices = list(current_indices) + queries
                query_sampler = SubsetRandomSampler(current_indices)
                query_dataloader = DataLoader(
                    train_dataset,
                    sampler=query_sampler,
                    batch_size=self.cfg.batch_size,
                    drop_last=True,
                )
                print(
                    "Sampled indices for next iteration. The new current indices will be: ",
                    current_indices,
                )

                # Optional: Save queries
                if self.cfg.save_queries:
                    if i == 0:
                        save_queries_dict = pd.DataFrame(
                            {"Iteration": [i], "Queries": [queries]}
                        )
                    else:
                        save_queries_dict = pd.concat(
                            [
                                save_queries_dict,
                                pd.DataFrame({"Iteration": [i], "Queries": [queries]}),
                            ],
                        )
                    save_queries_dict.to_csv(os.path.join(save_path, "queries.csv"))

    def result_handler(
        self, old_results, results, TaskLearner, AL_method, train_size, pool_size, n
    ):
        length = len(results["epoch"])
        results["TaskLearner"] = [TaskLearner] * length
        results["ActiveLearn"] = [AL_method] * length
        results["train_size"] = [train_size] * length
        results["pool_size"] = [pool_size] * length
        results["n"] = [n] * length

        results = pd.DataFrame.from_dict(results)

        if not isinstance(old_results, pd.DataFrame):
            return results

        results = pd.concat([old_results, results], ignore_index=True)

        return results

