import torch.nn.functional as F


def flow_matching_loss(
        model,
        t,
        x_0,
        x_1,
        elements,
        y,
        spg,
        n_sites,
        lattice_size: int = 3,    
        coords_loss_coef: float = 0.5,
        lattice_loss_coef: float = 0.5
):
    t = t.unsqueeze(1).unsqueeze(1)
    x_t = t * x_1 + (1 - t) * x_0
    t = t.flatten()

    output = model(x_t,
                   timesteps=t,
                   elements=elements,
                   y=y,
                   spg=spg)

    coords_truth, lattice_truth = (x_1 - x_0)[:, :-4], (x_1 - x_0)[:, -3:]
    coords_pred, lattice_pred = output[:, :-4], output[:, -3:]

    coords_loss = l1_loss(coords_truth.cpu(), coords_pred.cpu(), n_sites.cpu())
    lattice_loss = l1_loss(lattice_truth.cpu(), lattice_pred.cpu(), lattice_size)
    loss = coords_loss_coef * coords_loss + lattice_loss_coef * lattice_loss
    
    return loss