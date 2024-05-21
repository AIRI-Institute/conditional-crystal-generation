from collections import defaultdict
import json
import os
from pymatgen.symmetry.groups import SpaceGroup
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
import torch
import pandas as pd
from tqdm import tqdm
from ase.geometry import cellpar_to_cell, cell_to_cellpar

import re


def formula_to_list(formula):
    reg = re.findall("[A-Z][a-z]?|[0-9]+", formula)
    result = []
    for el, count in zip(reg[::2], reg[1::2]):
        result.extend([el] * int(count))
    return result


class CrystalDataset(torch.utils.data.Dataset):
    """
    Crystal dataset crystal data loads data from a pandas dataframe.
    For every item in the dataset it is needed to get the following data:
    - enthalpy_formation_atom
    - spacegroup
    - cell parameters
    - atom coordinates
    - atom types
    - nsites
    - element features
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        max_nsites=64,
        lattice_type="3_by_3",
        apply_energy_noising=False,
    ):
        """
        Args:
            dataframe: pandas.DataFrame or str
                dataframe containing the crystal data or a path to the dataframe
            max_nsites: int
                maximum number of sites in a crystal
            lattice_type: str "2_by_3" or "3_by_3"
                type of lattice to be calculated and concated to coordinates
        """
        super().__init__()
        self.py_utils_path = os.path.dirname(__file__)
        if type(dataframe) is str:
            dataframe = pd.read_csv(dataframe)

        self.dataframe = dataframe.copy()
        if apply_energy_noising:
            self.min_energy_deltas = self.dataframe.groupby("pretty_formula")[
                "enthalpy_formation_atom"
            ].apply(lambda group: np.diff(np.sort(group)).max())
            self.energy_noise = 0.01
        else:
            self.min_energy_deltas = None

        numeric_columns = self.dataframe.select_dtypes(include=["int", "float"])
        for col in numeric_columns.columns:
            self.dataframe[col] = self.dataframe[col].astype(np.float32)

        self.elm_onehot, self.elm_dict, self.elm_prop = self.get_one_hot()

        self.dataframe["elements"] = self.dataframe["pretty_formula"].apply(
            formula_to_list
        )

        self.dataframe["elements"] = self.dataframe["elements"].apply(
            lambda x: [self.elm_dict[j] for j in x]
        )

        self.dataframe["positions_fractional"] = self.dataframe[
            "positions_fractional"
        ].apply(eval)
        self.geometry = np.stack(
            self.dataframe["geometry"].apply(lambda x: np.array(eval(x))), axis=0
        ).astype(np.float32)

        self.dataframe = self.dataframe.drop(columns=["geometry"])
        self.column_to_index = dict(
            zip(self.dataframe.columns, range(len(self.dataframe.columns)))
        )
        self.data = self.dataframe.to_numpy()

        assert lattice_type in ["2_by_3", "3_by_3"]
        self.lattice_type = lattice_type

        self.max_nsites = max_nsites

        self.spg_to_tensor = {
            i: self.transrofm_spg_num_to_tensor(i) for i in range(1, 231)
        }

        if self.lattice_type == "3_by_3":
            self.geometry = [
                torch.from_numpy(cellpar_to_cell(i))
                for i in tqdm(self.geometry, desc="Converting lattice")
            ]

            self.geometry = torch.stack(self.geometry).float()
        else:
            self.geometry = torch.from_numpy(self.geometry).float()

    @staticmethod
    def get_3_by_3_lattice(lattice_2_by_3):
        a, b, c, alpha, beta, gamma = lattice_2_by_3

        ax, ay, az = a, 0, 0
        bx, by, bz = b * torch.cos(gamma), b * torch.sin(gamma), 0
        cx = c * torch.cos(alpha)
        cy = (
            c
            * (torch.cos(beta) - torch.cos(alpha) * torch.cos(gamma))
            / torch.sin(gamma)
        )
        cz = (
            c
            * torch.sqrt(
                1
                - torch.cos(beta) ** 2
                - torch.cos(alpha) ** 2
                - torch.cos(gamma) ** 2
                + 2 * torch.cos(beta) * torch.cos(alpha) * torch.cos(gamma)
            )
            / torch.sin(gamma)
        )

        matrix = torch.tensor([[ax, bx, cx], [ay, by, cy], [az, bz, cz]])
        return matrix

    @staticmethod
    def get_2_by_3_lattice(lattice_3_by_3):
        ax, bx, cx = lattice_3_by_3[0]
        ay, by, cy = lattice_3_by_3[1]
        az, bz, cz = lattice_3_by_3[2]

        a = torch.sqrt(ax**2 + ay**2 + az**2)
        b = torch.sqrt(bx**2 + by**2 + bz**2)
        c = torch.sqrt(cx**2 + cy**2 + cz**2)

        alpha = torch.acos((ax * bx + ay * by + az * cz) / (a * b)) * 180 / torch.pi
        beta = torch.acos((cx * bx + cy * by + cz * bz) / (c * b)) * 180 / torch.pi
        gamma = torch.acos((ax * cx + ay * cy + az * cz) / (a * c)) * 180 / torch.pi

        return torch.tensor((a, b, c, alpha, beta, gamma))

    @staticmethod
    def transrofm_spg_num_to_tensor(spg_num: int) -> torch.tensor:
        """
        This function transforms a space group number to a tensor with a shape (192, 4, 4).

        Inputs:
        spg_num: int
            Number of a space group

        Returns:
        affine_matrix_list: torch.tensor
            A tensor with all affine metrices of crystal symmetries.
            If a number of symmetries is less then 192, other part of affine_matrix_list is padded with zeros
        """
        affine_matrix_list = []

        symops = SpaceGroup.from_int_number(spg_num).symmetry_ops
        for op in symops:
            tmp = op.affine_matrix.astype(np.float32)

            if np.all(tmp == -1.0):
                print(tmp)
            affine_matrix_list.append(tmp)

        affine_matrix_list = np.array(affine_matrix_list)
        affine_matrix_list = np.vstack(
            (affine_matrix_list, np.zeros((192 - affine_matrix_list.shape[0], 4, 4)))
        )
        return torch.from_numpy(affine_matrix_list).float()

    def __getitem__(self, index):
        row = self.data[index]
        n_sites = int(row[self.column_to_index["nsites"]])
        new_order = np.arange(n_sites)

        elements = torch.tensor(row[self.column_to_index["elements"]])#[new_order]

        elemental_property_matrix = torch.vstack(
            (
                self.elm_prop[elements],
                torch.zeros((self.max_nsites - len(elements), self.elm_prop.shape[1])),
            )
        )

        elements = torch.vstack(
            (
                self.elm_onehot[elements],
                torch.zeros((self.max_nsites - len(elements), 103)),
            )
        ).float()

        lattice = self.geometry[index]
        coordinates = torch.tensor(row[self.column_to_index["positions_fractional"]])

        if self.lattice_type == "2_by_3":
            lattice = torch.reshape(lattice, (2, 3))

        coordinates_with_lattice = torch.vstack(
            (
                coordinates[new_order],
                torch.zeros((self.max_nsites - len(coordinates) - len(lattice), 3)),
                lattice,
            )
        )

        energy = np.float32(row[self.column_to_index["enthalpy_formation_atom"]])

        if self.min_energy_deltas is not None:
            energy += np.float32(
                np.random.uniform(-self.energy_noise / 2, self.energy_noise / 2)
            )

        results = {
            "spg": self.spg_to_tensor[row[self.column_to_index["spacegroup_relax"]]],
            "element_matrix": elements,
            "elemental_property_matrix": elemental_property_matrix,
            "coordinates_with_lattice": coordinates_with_lattice,
            "energy": energy,
            "n_sites": n_sites,
            "spacegroup_number": row[self.column_to_index["spacegroup_relax"]],
            "formula": row[self.column_to_index["pretty_formula"]],
        }
        return results

    def get_one_hot(self):
        self.list_of_elms = joblib.load(f"{self.py_utils_path}/../data/element.pkl")
        self.elm_dict = {element: i for i, element in enumerate(self.list_of_elms)}

        # Build one-hot vectors for the elements
        elm_onehot = np.arange(1, len(self.list_of_elms) + 1)[:, np.newaxis]
        elm_onehot = OneHotEncoder().fit_transform(elm_onehot).toarray()

        # property matrix
        with open(
            f"{self.py_utils_path}/../data/elemental_properties31-10-2023.json"
        ) as f:
            elm_prop = json.load(f)

        elm_prop = np.array(list(elm_prop.values()), dtype=float)
        elm_prop = np.nan_to_num(elm_prop, copy=False)
        elm_prop = torch.tensor(elm_prop, dtype=torch.float32)

        return torch.tensor(elm_onehot), self.elm_dict, elm_prop

    def __len__(self):
        return len(self.dataframe)
