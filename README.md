# conditional-crystal-generation

# Model Archetecture

UNet Archetecture            |  Condition Block
:---------------------------:|:-------------------------:
<img src="images/UNet_archetecture.jpg" alt="drawing" width="400"/>|<img src="images/condition_block_archetecture.jpg" alt="drawing" width="400"/>


# Repository structure
```
|── notebooks
│   ├── diffusion_generation_inference.ipynb
│   ├── diffusion_generation_train.ipynb
│   ├── diffusion_modification_train.ipynb
│   ├── flow_matching_generation_inference.ipynb
│   ├── flow_matching_generation_train.ipynb
│   └── flow_matching_modification_train.ipynb
├── requirements.txt
└── src
    ├── data
    │   ├── element.pkl
    │   └── elemental_properties31-10-2023.json
    ├── generation
    │   ├── diffusion_generation_loops.py
    │   ├── flow_matching_generation_loops.py
    │   ├── generation.py
    │   └── regression_generation_loops.py
    ├── inference
    │   └── inference_data_generation.py
    ├── losses.py
    ├── model
    │   ├── fp16_util.py
    │   ├── models.py
    │   ├── nn.py
    │   └── unet.py
    ├── modification
    │   ├── diffusion_modification_loops.py
    │   ├── flow_matching_modification_loops.py
    │   ├── modification.py
    │   └── regression_modification_loops.py
    ├── py_utils
    │   ├── comparator.py
    │   ├── crystal_dataset.py
    │   ├── sampler.py
    │   ├── skmultilearn_iterative_split.py
    │   └── stratified_splitter.py
    └── utils.py

```
