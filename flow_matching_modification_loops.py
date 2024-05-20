import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm

from generation import generate_flow_matching
from losses import flow_matching_loss
from py_utils.comparator import PymatgenComparator


def train_epoch(
    model,
    optimizer,
    loss_function: flow_matching_loss,
    coords_loss_coef: float,
    lattice_loss_coef: float,
    train_dataloader: torch.utils.data.DataLoader,
    scheduler,
    accelerator: Accelerator,
    lattice_size: int = 3,
    device: str = "cuda",
):
    # -------------------------------------------------------------#
    # Train epoch
    total_train_loss = 0.0

    model.train()
    for batch in tqdm(train_dataloader):
        with accelerator.accumulate(model):
            x_0_coords = batch["x0_coordinates_with_lattice"]
            x_1_coords = batch["x1_coordinates_with_lattice"]
            element_matrix = batch["element_matrix"]
            elemental_property_matrix = batch["elemental_property_matrix"]
            spg = batch["spg"]
            condition = batch["energy"]
            n_sites = batch["n_sites"]
            (
                x_0_coords,
                x_1_coords,
                element_matrix,
                elemental_property_matrix,
                spg,
                condition,
                n_sites,
            ) = (
                x_0_coords.to(device),
                x_1_coords.to(device),
                element_matrix.to(device),
                elemental_property_matrix.to(device),
                spg.to(device),
                condition.to(device),
                n_sites.to(device),
            )

            elements = torch.cat([element_matrix, elemental_property_matrix], dim=-1)

            # time t
            t = torch.rand(x_1_coords.shape[0]).to(device)

            optimizer.zero_grad()
            coords_loss, lattice_loss, loss = loss_function(
                model=model,
                t=t,
                x_0=x_0_coords,
                x_1=x_1_coords,
                elements=elements,
                y=condition,
                spg=spg,
                n_sites=n_sites,
                lattice_size=lattice_size,
                coords_loss_coef=coords_loss_coef,
                lattice_loss_coef=lattice_loss_coef,
            )

        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()

    train_dict = {
        "train_manhattan_loss": total_train_loss / len(train_dataloader),
    }

    return train_dict


def eval_epoch(
    model,
    loss_function,
    comparator: PymatgenComparator,
    eval_dataloader: torch.utils.data.DataLoader,
    lattice_size: int = 3,
    device: str = "cuda",
):
    # -------------------------------------------------------------#
    # Eval epoch
    compares_metrics = []
    test_atomic_metrics = []
    test_lattice_metrics = []

    model.eval()
    for batch in tqdm(eval_dataloader):
        x_0_coords = batch["x0_coordinates_with_lattice"]
        x_1_coords = batch["x1_coordinates_with_lattice"]
        element_matrix = batch["element_matrix"]
        elemental_property_matrix = batch["elemental_property_matrix"]
        spg = batch["spg"]
        condition = batch["energy"]
        n_sites = batch["n_sites"]
        (
            x_0_coords,
            x_1_coords,
            element_matrix,
            elemental_property_matrix,
            spg,
            condition,
            n_sites,
        ) = (
            x_0_coords.to(device),
            x_1_coords.to(device),
            element_matrix.to(device),
            elemental_property_matrix.to(device),
            spg.to(device),
            condition.to(device),
            n_sites.to(device),
        )

        elements = torch.cat([element_matrix, elemental_property_matrix], dim=-1)

        x_1_gen = generate_flow_matching(
            model=model,
            x_0=x_0_coords,
            elements=elements,
            y=condition,
            spg=spg,
            device=device,
        )

        coords_truth, lattice_truth = x_1_coords[:, :-4], x_1_coords[:, -3:]
        coords_pred, lattice_pred = x_1_gen[:, :-4], x_1_gen[:, -3:]

        test_atomic_metric = loss_function(
            coords_pred.cpu(), coords_truth.cpu(), n_sites.cpu()
        )
        test_lattice_metric = loss_function(
            lattice_pred.cpu(), lattice_truth.cpu(), lattice_size
        )

        batch_size = x_0_coords.shape[0]
        test_atomic_metrics.append(test_atomic_metric.item() / batch_size)
        test_lattice_metrics.append(test_lattice_metric.item() / batch_size)

        compares = comparator.calculate_compares(
            element_matrix,
            n_sites,
            coords_truth,
            lattice_truth,
            coords_pred,
            lattice_pred,
        )
        batch_size = x_0_coords.shape[0]
        compares_metrics.append(compares.sum(axis=1) / batch_size)

    eval_dict = {
        "val_atomic_euclidean_loss": np.mean(test_atomic_metrics),
        "val_lattice_euclidean_loss": np.mean(test_lattice_metrics),
        "val_metric_default": np.mean(compares_metrics, axis=0)[0],
        "val_metric_defaultX2": np.mean(compares_metrics, axis=0)[1],
        "val_metric_defaultX5": np.mean(compares_metrics, axis=0)[2],
    }

    return eval_dict


def train(
    model: torch.nn.Module,
    optimizer,
    loss_function,
    metric_function,
    comparator,
    coords_loss_coef: float,
    lattice_loss_coef: float,
    epochs: int,
    train_dataloader: torch.utils.data.DataLoader,
    eval_dataloader: torch.utils.data.DataLoader,
    scheduler,
    accelerator: Accelerator,
    lattice_size: int = 3,
    device: str = "cuda",
    eval_every_n: int = 5,
):
    for i in tqdm(range(epochs)):
        train_logs = train_epoch(
            model,
            optimizer,
            loss_function,
            coords_loss_coef,
            lattice_loss_coef,
            train_dataloader,
            scheduler,
            accelerator,
            lattice_size,
            device,
        )

        if i % eval_every_n == 0:
            eval_logs = eval_epoch(
                model,
                metric_function,
                comparator,
                eval_dataloader,
                lattice_size,
                device,
            )

            train_logs.update(eval_logs)

        print(train_logs)
