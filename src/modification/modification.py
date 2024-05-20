import torch


def modify_flow_matching(model, x0, elements, y, spg, device: str = "cuda"):
    model.eval()
    model.to(device)

    xt = x0
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


def modify_regressor(model, x0, elements, y, spg, device: str = "cuda"):
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

def modify_diffusion(
        x, 
        model, 
        x_0,
        elements, 
        condition, 
        spg, 
        noise_scheduler, 
        num_inference_steps: int = 100
    ):
    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    for i, t in enumerate(noise_scheduler.timesteps):
        model_input = noise_scheduler.scale_model_input(x, t)

        t_batch = torch.full(
            size=(x.shape[0],), 
            fill_value=t.item(), 
            dtype=torch.long
        ).cuda()

        with torch.no_grad():
            noise_pred = model(
                x=model_input, 
                x_0=x_0,
                timesteps=t_batch,
                spg=spg,
                y=condition, 
                elements=elements, 
            )

        x = noise_scheduler.step(noise_pred, t, x).prev_sample

    return x
