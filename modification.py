import torch


def modify_flow_matching(model, x0, elements, y, spg, device: str = "cuda"):
    model.eval()
    model.to(device)

    xt = x0
    xt, elements, y, spg = xt.to(device), elements.to(device), y.to(device), spg.to(device)

    eps = 1e-8
    n_steps = 100
    t = torch.linspace(eps, 1 - eps, n_steps + 1).to(xt.device)

    for i in range(1, len(t)):
        with torch.no_grad():
            t_prev = t[i - 1].unsqueeze(0)

            f_eval = model(
                xt,
                timesteps=t_prev,
                y=y,
                elements=elements,
                spg=spg
            )
        x = xt + (t[i] - t[i - 1]) * f_eval
        xt = x

    return xt