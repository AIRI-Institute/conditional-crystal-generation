import torch
import tqdm
from src.losses import l1_loss, pbc_l1_loss
import numpy as np
from src.py_utils.comparator import PymatgenComparator


def train_epoch(
    model: torch.nn.Module,
    optimizer,
    atomic_loss_fn: l1_loss,
    lattice_loss_fn: pbc_l1_loss,
    metric: pbc_l1_loss,
    coords_loss_coef: float,
    lattice_loss_coef: float,
    train_dataloader: torch.utils.data.DataLoader,
    scheduler,
    lattice_size: int = 3,
    device: str = "cuda",
):
    train_losses = []
    train_atomic_metrics = []
    train_lattice_metrics = []
    # Train epoch
    model.train()
    for batch in tqdm(train_dataloader):
        # get needed features
        x_0_coords = batch["x0_coordinates_with_lattice"]
        x_1_coords = batch["x1_coordinates_with_lattice"]
        element_matrix = batch["element_matrix"]
        elemental_property_matrix = batch["elemental_property_matrix"]
        spg = batch["spg"]
        condition = batch["energy"]
        n_sites = batch["nsites"]
        formulas = batch["formula"]

        (
            x_0_coords,
            x_1_coords,
            element_matrix,
            elemental_property_matrix,
            spg,
            condition,
        ) = (
            x_0_coords.to(device),
            x_1_coords.to(device),
            element_matrix.to(device),
            elemental_property_matrix.to(device),
            spg.to(device),
            condition.to(device),
        )

        optimizer.zero_grad()
        output = model(
            x_0_coords,
            elements=torch.cat([element_matrix, elemental_property_matrix], dim=-1),
            y=condition,
            spg=spg,
        )

        x_1_coords = x_1_coords.cpu()
        output = output.cpu()

        coords_truth, lattice_truth = (
            x_1_coords[:, :-4],
            x_1_coords[:, -lattice_size:],
        )
        coords_pred, lattice_pred = output[:, :-4], output[:, -lattice_size:]

        coords_loss = atomic_loss_fn(coords_truth, coords_pred, formulas)
        lattice_loss = lattice_loss_fn(lattice_truth, lattice_pred, lattice_size)
        train_loss = coords_loss_coef * coords_loss + lattice_loss_coef * lattice_loss

        train_atomic_metric = metric(coords_pred, coords_truth, n_sites)
        train_lattice_metric = metric(lattice_pred, lattice_truth, lattice_size)

        train_loss.backward()
        optimizer.step()
        scheduler.step()

        batch_size = x_0_coords.shape[0]
        train_losses.append(train_loss.item() / batch_size)
        train_atomic_metrics.append(train_atomic_metric.item() / batch_size)
        train_lattice_metrics.append(train_lattice_metric.item() / batch_size)

    return train_losses, train_atomic_metrics, train_lattice_metrics


def eval_epoch(
    model: torch.nn.Module,
    atomic_loss_fn: l1_loss,
    lattice_loss_fn: pbc_l1_loss,
    metric: pbc_l1_loss,
    coords_loss_coef: float,
    lattice_loss_coef: float,
    test_dataloader: torch.utils.data.DataLoader,
    comparator: PymatgenComparator,
    lattice_size: int = 3,
    device: str = "cuda",
):
    test_losses = []
    test_atomic_metrics = []
    test_lattice_metrics = []
    compares_metrics = []

    model.eval()
    for batch in tqdm(test_dataloader):
        # get needed features
        x_0_coords = batch["x0_coordinates_with_lattice"]
        x_1_coords = batch["x1_coordinates_with_lattice"]
        element_matrix = batch["element_matrix"]
        elemental_property_matrix = batch["elemental_property_matrix"]
        spg = batch["spg"]
        formulas = batch["formula"]

        condition = batch["energy"]
        n_sites = batch["nsites"]
        (
            x_0_coords,
            x_1_coords,
            element_matrix,
            elemental_property_matrix,
            spg,
            condition,
        ) = (
            x_0_coords.to(device),
            x_1_coords.to(device),
            element_matrix.to(device),
            elemental_property_matrix.to(device),
            spg.to(device),
            condition.to(device),
        )

        with torch.no_grad():
            output = model(
                x_0_coords,
                elements=torch.cat([element_matrix, elemental_property_matrix], dim=-1),
                y=condition,
                spg=spg,
            )

            x_1_coords = x_1_coords.cpu()
            output = output.cpu()

            coords_truth, lattice_truth = (
                x_1_coords[:, :-4],
                x_1_coords[:, -lattice_size:],
            )
            coords_pred, lattice_pred = output[:, :-4], output[:, -lattice_size:]

            coords_loss = atomic_loss_fn(coords_truth, coords_pred, formulas)
            lattice_loss = lattice_loss_fn(lattice_truth, lattice_pred, lattice_size)
            test_loss = (
                coords_loss_coef * coords_loss + lattice_loss_coef * lattice_loss
            )

            test_atomic_metric = metric(coords_pred, coords_truth, n_sites)
            test_lattice_metric = metric(lattice_pred, lattice_truth, lattice_size)

        batch_size = x_0_coords.shape[0]

        compares = comparator.calculate_compares(
            element_matrix,
            n_sites,
            coords_truth,
            lattice_truth,
            coords_pred,
            lattice_pred,
        )
        compares_metrics.append(compares.sum(axis=1) / batch_size)

        test_losses.append(test_loss.item() / batch_size)
        test_atomic_metrics.append(test_atomic_metric.item() / batch_size)
        test_lattice_metrics.append(test_lattice_metric.item() / batch_size)

    return test_losses, test_atomic_metrics, test_lattice_metrics, compares_metrics


def train(
    model: torch.nn.Module,
    optimizer,
    atomic_loss_fn,
    lattice_loss_fn,
    metric,
    coords_loss_coef: float,
    lattice_loss_coef: float,
    epochs: int,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    scheduler,
    lattice_size: int = 3,
    device: str = "cuda",
):

    for epoch in range(epochs):
        train_losses, train_atomic_metrics, train_lattice_metrics = train_epoch(
            model,
            optimizer,
            atomic_loss_fn,
            lattice_loss_fn,
            metric,
            coords_loss_coef,
            lattice_loss_coef,
            train_dataloader,
            scheduler,
            lattice_size=lattice_size,
            device=device,
        )
        (
            test_losses,
            test_atomic_metrics,
            test_lattice_metrics,
            test_compares_metrics,
        ) = eval_epoch(
            model,
            atomic_loss_fn,
            lattice_loss_fn,
            metric,
            coords_loss_coef,
            lattice_loss_coef,
            test_dataloader,
            PymatgenComparator(),
            lattice_size=lattice_size,
            device=device,
        )
        print(
            {
                "epoch": epoch + 1,
                "train_atomic_euclidean_loss": np.mean(train_atomic_metrics),
                "val_atomic_euclidean_loss": np.mean(test_atomic_metrics),
                "train_lattice_euclidean_loss": np.mean(train_lattice_metrics),
                "val_lattice_euclidean_loss": np.mean(test_lattice_metrics),
                "train_manhattan_loss": np.mean(train_losses),
                "val_manhattan_loss": np.mean(test_losses),
                "val_metric_default": np.mean(test_compares_metrics),
            }
        )
