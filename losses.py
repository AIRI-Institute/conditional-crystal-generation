import torch
import torch.nn.functional as F
from torch.linalg import norm


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
    lattice_loss_coef: float = 0.5,
):
    t = t.unsqueeze(1).unsqueeze(1)
    x_t = t * x_1 + (1 - t) * x_0
    t = t.flatten()

    output = model(x_t, timesteps=t, elements=elements, y=y, spg=spg)

    coords_truth, lattice_truth = (x_1 - x_0)[:, :-4], (x_1 - x_0)[:, -3:]
    coords_pred, lattice_pred = output[:, :-4], output[:, -3:]

    coords_loss = l1_loss(coords_truth.cpu(), coords_pred.cpu(), n_sites.cpu())
    lattice_loss = l1_loss(lattice_truth.cpu(), lattice_pred.cpu(), lattice_size)
    loss = coords_loss_coef * coords_loss + lattice_loss_coef * lattice_loss

    return coords_loss, lattice_loss, loss

def diffsion_generation_loss(
                model,
                t,
                noise,
                noisy_x_1,
                elements,
                y,
                spg,
                n_sites,
                lattice_size: int = 3,
                coords_loss_coef: float = 0.5,
                lattice_loss_coef: float = 0.5,
            ):
    
    output = model(
                noisy_x_1, 
                elements=elements, 
                y=y, 
                spg=spg, 
                timesteps=t
            )
                
    coords_loss = l1_loss(noise[:, :-4], output[:, :-4], n_sites)
    lattice_loss = l1_loss(noise[:, -3:], output[:, -3:], lattice_size)
    
    train_loss = coords_loss_coef * coords_loss + lattice_loss_coef * lattice_loss

    return coords_loss, lattice_loss, train_loss

def diffsion_modification_loss(
                model,
                t,
                noise,
                noisy_x_1,
                x_0,
                elements,
                y,
                spg,
                n_sites,
                lattice_size,
                coords_loss_coef,
                lattice_loss_coef,
            ):
    output = model(
                noisy_x_1, 
                elements=elements, 
                y=y, 
                spg=spg, 
                timesteps=t,
                x_0=x_0
            )
                
    coords_loss = l1_loss(noise[:, :-4], output[:, :-4], n_sites)
    lattice_loss = l1_loss(noise[:, -3:], output[:, -3:], lattice_size)
    
    train_loss = coords_loss_coef * coords_loss + lattice_loss_coef * lattice_loss

    return coords_loss, lattice_loss, train_loss


def l1_loss(
    predicted_features,  # [batch_size, n_sites, 3]
    target_features,  # [batch_size, n_sites, 3]
    n_sites=None,  # [batch_size, ]
):
    loss = norm(predicted_features - target_features, ord=1, dim=-1).sum(-1)
    # loss = torch.abs(predicted_features - target_features).sum(-1).sum(-1)
    if n_sites is None:
        n_sites = 1
    loss = loss / n_sites
    loss = loss.sum()
    return loss


def vertex_target_mask(target):
    bin_mask = ((target == 1.0) | (target == 0.0)).sum(dim=2) == 3
    bin_mask = bin_mask.unsqueeze(-1).repeat((1, 1, 3))
    return bin_mask


def vertex_min_norm(preds, target, norm_ord=2):
    mask = vertex_target_mask(target)

    if mask.sum() == 0:
        return False

    masked_preds = preds * mask
    batch_size, sites_num, _ = preds.shape

    dists = torch.zeros((batch_size, sites_num, 8))
    vertices = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]
    for i, vertex in enumerate(vertices):
        vertex_target = (
            torch.tensor(vertex).reshape(1, 1, 3).repeat((batch_size, sites_num, 1))
        )
        dist = norm(masked_preds - vertex_target, dim=-1, ord=norm_ord)
        dists[:, :, i] = dist

    return norm(dists, dim=-1, ord=-torch.inf)  # differentiable min function


def edge_target_mask(target, first_true_index, second_true_index, false_index):
    bin_mask = (target == 1.0) | (target == 0.0)
    fixed_index_mask = (
        (bin_mask[:, :, first_true_index] == bin_mask[:, :, second_true_index])
        & bin_mask[:, :, first_true_index]
        & ~bin_mask[:, :, false_index]
    )

    edge_coords_mask = fixed_index_mask.unsqueeze(-1).repeat((1, 1, 3))
    edge_coords_mask[:, :, false_index] = False

    fixed_points_mask = fixed_index_mask.unsqueeze(-1).repeat((1, 1, 3))
    fixed_points_mask[:, :, (first_true_index, second_true_index)] = False

    error_mask = fixed_index_mask

    return edge_coords_mask, fixed_points_mask, error_mask


def edge_min_norm(
    preds, target, first_true_index=0, second_true_index=1, false_index=2, norm_ord=2
):
    edge_coords_mask, fixed_points_mask, error_mask = edge_target_mask(
        target, first_true_index, second_true_index, false_index
    )
    if error_mask.sum() == 0:
        return 0

    if false_index == 0:
        vertices = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]
    elif false_index == 1:
        vertices = [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]]
    if false_index == 2:
        vertices = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]

    batch_size, sites_num, _ = preds.shape

    dists = torch.zeros((batch_size, sites_num, 4))

    for i, vertex in enumerate(vertices):
        vertex_target = (
            torch.tensor(vertex).reshape(1, 1, 3).repeat((batch_size, sites_num, 1))
        )
        vertex_target = edge_coords_mask * vertex_target + fixed_points_mask * target

        dist = norm(preds - vertex_target, dim=-1, ord=norm_ord)
        dists[:, :, i] = dist

    min_dists = norm(dists, dim=-1, ord=-torch.inf)

    return min_dists * error_mask  # differentiable min function


def side_target_mask(target, true_index=0):
    false_indexes = [0, 1, 2]
    false_indexes.remove(true_index)
    first_false_idx, second_false_idx = false_indexes

    bin_mask = (target == 1.0) | (target == 0.0)
    fixed_index_mask = (
        (bin_mask[:, :, first_false_idx] == bin_mask[:, :, second_false_idx])
        & ~bin_mask[:, :, first_false_idx]
        & bin_mask[:, :, true_index]
    )

    side_coords_mask = fixed_index_mask.unsqueeze(-1).repeat((1, 1, 3))
    side_coords_mask[:, :, false_indexes] = False

    fixed_points_mask = fixed_index_mask.unsqueeze(-1).repeat((1, 1, 3))
    fixed_points_mask[:, :, true_index] = False

    error_mask = fixed_index_mask

    return side_coords_mask, fixed_points_mask, error_mask


def side_min_norm(preds, target, true_index=0, norm_ord=2):
    side_coords_mask, fixed_points_mask, error_mask = side_target_mask(
        target, true_index
    )

    if error_mask.sum() == 0:
        return 0

    batch_size, sites_num, _ = preds.shape

    dists = torch.zeros((batch_size, sites_num, 2))

    for i in range(2):
        vertex_target = side_coords_mask * i + fixed_points_mask * target
        dist = norm(preds - vertex_target, dim=-1, ord=norm_ord)
        dists[:, :, i] = dist

    min_dists = norm(dists, dim=-1, ord=-torch.inf)

    return min_dists * error_mask  # differentiable min function


def default_point_norm(preds, target, norm_ord=2):
    bin_mask = ((target == 1.0) | (target == 0.0)).sum(dim=2) == 0
    bin_mask = bin_mask.unsqueeze(-1).repeat((1, 1, 3))

    masked_target = target * bin_mask
    masked_preds = preds * bin_mask
    return norm(masked_preds - masked_target, dim=-1, ord=norm_ord)


def calc_total_dists(preds, target, norm_ord=2):
    total_dists = 0
    total_dists += default_point_norm(preds, target, norm_ord)
    total_dists += vertex_min_norm(preds, target, norm_ord)

    for true_idx_1, true_idx_2, false_idx in ([0, 1, 2], [0, 2, 1], [1, 2, 0]):
        total_dists += edge_min_norm(
            preds, target, true_idx_1, true_idx_2, false_idx, norm_ord
        )

    for true_index in range(3):
        total_dists += side_min_norm(preds, target, true_index)

    return total_dists


def pbc_l1_loss(preds, target, n_sites=None):
    total_dists = calc_total_dists(preds, target, norm_ord=1).sum(-1)
    if n_sites is None:
        n_sites = 1
    total_dists /= n_sites
    return total_dists.sum()
