import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm

from py_utils.loss_and_metrics import PymatgenComparator
from losses import diffsion_generation_loss
from generation import generate_diffusion


def train_epoch(
    model,
    optimizer,
    noise_scheduler,
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
        # get needed features
        x_1_coords  = batch["coordinates_with_lattice"]
        element_matrix  = batch["element_matrix"]
        elemental_property_matrix  = batch["elemental_property_matrix"]
        spg = batch["spg"]
        condition = batch["energy"]
        n_sites = batch["n_sites"]
        (
            x_1_coords,
            element_matrix,
            elemental_property_matrix,
            spg,
            condition, 
            n_sites
            ) = (
                    x_1_coords.to(device), 
                    element_matrix.to(device), 
                    elemental_property_matrix.to(device), 
                    spg.to(device), 
                    condition.to(device), 
                    n_sites.to(device)
                )
        bs = x_1_coords.shape[0]
        noise = torch.randn(x_1_coords.shape).to(x_1_coords.device)
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, (bs,), device=x_1_coords.device
        ).long()

        noisy_x_1_coords = noise_scheduler.add_noise(x_1_coords, noise, timesteps)
        elements = torch.cat([element_matrix, elemental_property_matrix], dim=-1)

        
        with accelerator.accumulate(model):
            
            coords_loss, lattice_loss, loss = diffsion_generation_loss(
                model=model,
                t=timesteps,
                noise=noise,
                noisy_x_1=noisy_x_1_coords,
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
            optimizer.zero_grad()

            total_train_loss += loss.item() / bs

    train_dict = {
        "train_manhattan_loss": total_train_loss / len(train_dataloader),
    }

    return train_dict


def eval_epoch(
    model,
    noise_scheduler,
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
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            x_1_coords = batch["coordinates_with_lattice"]
            element_matrix  = batch["element_matrix"]
            elemental_property_matrix  = batch["elemental_property_matrix"]
            spg = batch["spg"]
            condition = batch["energy"]
            n_sites = batch["n_sites"]
            (
                x_1_coords,
                element_matrix,
                elemental_property_matrix,
                spg,
                condition, 
                n_sites
            ) = (
                    x_1_coords.to(device), 
                    element_matrix.to(device), 
                    elemental_property_matrix.to(device), 
                    spg.to(device), 
                    condition.to(device), 
                    n_sites.to(device)
                )
            
            elements = torch.cat([element_matrix, elemental_property_matrix], dim=-1)

            noise = torch.randn(x_1_coords.shape).to(device) 
            
            output = generate_diffusion(
                x=noise, 
                model=model, 
                elements=elements, 
                condition=condition, 
                spg=spg, 
                noise_scheduler=noise_scheduler,
            )

                
            coords_truth, lattice_truth = x_1_coords[:, :-4], x_1_coords[:, -3:]
            coords_pred, lattice_pred = output[:, :-4], output[:, -3:]
            
            test_atomic_metric = loss_function(coords_pred.cpu(), coords_truth.cpu(), n_sites.cpu())
            test_lattice_metric = loss_function(lattice_pred.cpu(), lattice_truth.cpu(), lattice_size)

            batch_size = x_1_coords.shape[0]
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
            batch_size = x_1_coords.shape[0]
            compares_metrics.append(compares.sum(axis=1) / batch_size)
        
        eval_dict = {
            "epoch": epoch + 1,
            
            "val_atomic_euclidean_loss": np.mean(test_atomic_metrics),
        
            "val_lattice_euclidean_loss": np.mean(test_lattice_metrics),
            
            "val_metric_default": np.mean(compares_metrics, axis=0)[0],
        }

    return eval_dict


def train(
    model: torch.nn.Module,
    optimizer,
    noise_scheduler,
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
            noise_scheduler,
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
                noise_scheduler,
                metric_function,
                comparator,
                eval_dataloader,
                lattice_size,
                device,
            )

            train_logs.update(eval_logs)

        print(train_logs)