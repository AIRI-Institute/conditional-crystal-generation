import torch


def generate_flow_matching(model, x_0, elements, y, spg, device: str = "cuda"):
    model.eval()
    model.to(device)

    xt = x_0
    xt, elements, y, spg = (
        xt.to(device),
        elements.to(device),
        y.to(device),
        spg.to(device),
    )

    eps = 1e-8
    n_steps = 100
    t = torch.linspace(eps, 1 - eps, n_steps + 1).to(xt.device)

    for i in range(1, len(t)):
        with torch.no_grad():
            t_prev = t[i - 1].unsqueeze(0)
            f_eval = model(xt, timesteps=t_prev, y=y, elements=elements, spg=spg)

        x = xt + (t[i] - t[i - 1]) * f_eval
        xt = x

    return xt


def generate_regressor(model, x0, elements, y, spg, device: str = "cuda"):
    model.eval()
    model.to(device)

    x0, elements, y, spg = (
        x0.to(device),
        elements.to(device),
        y.to(device),
        spg.to(device),
    )
    with torch.no_grad():
        output = model(
            x0,
            elements=elements,
            y=y,
            spg=spg,
        )

        output = output.cpu()

    return output



def generate_diffusion(
        x_0,
        model, 
        elements, 
        y,
        spg, 
        noise_scheduler, 
        num_inference_steps: int = 100,
        device: str = "cuda"
    ):
    model.eval()
    model.to(device)

    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    xt = x_0
    xt, elements, y, spg = (
        xt.to(device),
        elements.to(device),
        y.to(device),
        spg.to(device),
    )

    for i, t in enumerate(noise_scheduler.timesteps):
        xt = noise_scheduler.scale_model_input(xt, t)

        t_batch = torch.full(
            size=(x.shape[0],), 
            fill_value=t.item(), 
            dtype=torch.long
        ).cuda()

        with torch.no_grad():
            noise_pred = model(
                x=xt,
                timesteps=t_batch,
                spg=spg,
                y=y,
                elements=elements, 
            )

        xt = noise_scheduler.step(noise_pred, t, xt).prev_sample

    return xt