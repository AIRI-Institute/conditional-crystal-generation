import torch
from torch import nn

from model.unet import UNetModel


class CrystalUNetModel(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            dims,
            condition_dims,
            model_channels,
            num_res_blocks,
            attention_resolutions=(1, 2, 4, 8)
    ):
        super(CrystalUNetModel, self).__init__()

        self.model = UNetModel(
            in_channels=in_channels,  # should be equal to num_features (input features)
            dims=dims,  # this states, that we are using 1D U-Net
            condition_dims=condition_dims,  # num_condition_features
            model_channels=model_channels,  # inner model features
            out_channels=out_channels,  # should be equal to num_features (input features)
            num_res_blocks=num_res_blocks,  # idk
            attention_resolutions=attention_resolutions  # idk
        )

        self.spg_condition = nn.Sequential(
            nn.Conv2d(192, 64, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Flatten()
        )

        self.element_condition = nn.Sequential(
            nn.Conv1d(125, 64, 3, 1),
            nn.GELU(),

            nn.Conv1d(64, 128, 3, 1),
            nn.GELU(),

            nn.Conv1d(128, 256, 3, 1),
            nn.GELU(),

            nn.Flatten(),

            nn.Linear(14848, 256)
        )

    def forward(
            self,
            x,
            elements,
            y,
            spg,
            timesteps=None
    ):
        spg = self.spg_condition(spg)
        elements = self.element_condition(elements.permute(0, 2, 1))
        x = self.model(
            x=x,
            timesteps=timesteps,
            y=torch.cat((y.unsqueeze(dim=-1), spg, elements), dim=1)

        )
        return x
    
class ModelSPGCondition(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels,
        dims,
        condition_dims,
        model_channels,
        num_res_blocks,
        attention_resolutions=("16",)
    ):
        super(ModelSPGCondition, self).__init__()
    
        self.model = UNetModel(
            in_channels=in_channels, # should be equal to num_features (input features) 
            dims=dims, #this states, that we are using 1D U-Net
            condition_dims=condition_dims, # num_condition_features
            model_channels=model_channels, # inner model features
            out_channels=out_channels, # should be equal to num_features (input features) 
            num_res_blocks=num_res_blocks, # idk
            attention_resolutions=attention_resolutions # idk
        )

        self.spg_condition = nn.Sequential(
            nn.Conv2d(192, 64, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Flatten()
        )
        
        self.element_condition = nn.Sequential(
            nn.Conv1d(125, 64, 3, 1),
            nn.GELU(),

            nn.Conv1d(64, 128, 3, 1),
            nn.GELU(),

            nn.Conv1d(128, 256, 3, 1),
            nn.GELU(),

            nn.Flatten(),
    
            nn.Linear(14848, 256)
        )

        self.coords_condition = nn.Sequential(
            nn.Conv1d(64, 128, 2, 1),
            nn.GELU(),

            nn.Conv1d(128, 256, 2, 1),
            nn.GELU(),

            nn.Flatten(),
    
            nn.Linear(256, 256)
        )
    

    def forward(
        self, 
        x, 
        elements,
        y, 
        spg, 
        x_0_coords,
        timesteps=None
    ):
        x_0_coords = self.coords_condition(x_0_coords)
        spg = self.spg_condition(spg)
        elements = self.element_condition(elements.permute(0, 2, 1))
        x = self.model(
            x=x, 
            timesteps=timesteps,
            y=torch.cat((y.unsqueeze(dim=-1), spg, elements, x_0_coords), dim=1)
        )
        return x
