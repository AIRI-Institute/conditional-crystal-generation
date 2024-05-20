import numpy as np
from typing import Union, Tuple

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

import re


def count_atoms(formula):
    return sum(list(map(int, re.findall(r'\d+', formula))))


def formula_to_list(formula):
    reg = re.findall("[A-Z][a-z]?|[0-9]+", formula)
    result = []
    for el, count in zip(reg[::2], reg[1::2]):
        result.extend([el] * int(count))
    return result


class InferenceCrystalDataset(torch.utils.data.Dataset):
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
            dataframe: Union[pd.DataFrame, str],
            max_nsites: int = 64,
            lattice_type: str = "3_by_3",
            random_shuffle_coordinates: bool = False,
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
        self.random_shuffle_coordinates = random_shuffle_coordinates

        if type(dataframe) is str:
            dataframe = pd.read_csv(dataframe)

        self.dataframe = dataframe.copy()

        self.dataframe["nsites"] = self.dataframe["pretty_formula"].apply(count_atoms)

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

    def __getitem__(self, index, shuffle_order=None):
        row = self.data[index]
        n_sites = int(row[self.column_to_index["nsites"]])
        # print(n_sites)

        if self.random_shuffle_coordinates and shuffle_order is None:
            new_order = np.random.choice(
                np.arange(n_sites), size=n_sites, replace=False
            )
        elif self.random_shuffle_coordinates and shuffle_order is not None:
            new_order = shuffle_order
        else:
            new_order = np.arange(n_sites)

        elements = torch.tensor(row[self.column_to_index["elements"]])[new_order]

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

        energy = np.float32(row[self.column_to_index["enthalpy_formation_atom"]])

        results = {
            "spg": self.spg_to_tensor[row[self.column_to_index["spacegroup_relax"]]],
            "element_matrix": elements,
            "elemental_property_matrix": elemental_property_matrix,
            "energy": energy,
            "n_sites": n_sites,
            "spacegroup_number": row[self.column_to_index["spacegroup_relax"]],
        }
        return results

    def get_one_hot(self):
        self.list_of_elms = joblib.load(f"../data/element.pkl")
        self.elm_dict = {element: i for i, element in enumerate(self.list_of_elms)}

        # Build one-hot vectors for the elements
        elm_onehot = np.arange(1, len(self.list_of_elms) + 1)[:, np.newaxis]
        elm_onehot = OneHotEncoder().fit_transform(elm_onehot).toarray()

        # property matrix
        with open(
                f"../data/elemental_properties31-10-2023.json"
        ) as f:
            elm_prop = json.load(f)

        elm_prop = np.array(list(elm_prop.values()), dtype=float)
        elm_prop = np.nan_to_num(elm_prop, copy=False)
        elm_prop = torch.tensor(elm_prop, dtype=torch.float32)

        return torch.tensor(elm_onehot), self.elm_dict, elm_prop

    def __len__(self):
        return len(self.dataframe)


def generate_inference_dataset(
        formula,
        spgs,
        step,
        start,
        n,
        return_df: bool = False,
        max_nsites: int = 64,
        lattice_type: str = "3_by_3",
        random_shuffle_coordinates: bool = False
) -> Union[InferenceCrystalDataset, Tuple[pd.DataFrame, InferenceCrystalDataset]]:
    energies = np.arange(start, start + ((n + 1) * step), step)

    num_spgs = len(spgs)
    num_energies = len(energies)

    spg_comb = []
    energy_comb = []

    for i in range(num_spgs):
        spg_comb.extend([spgs[i]] * num_energies)
        energy_comb.extend(energies)

    formula_comb = [formula] * len(energy_comb)

    df = pd.DataFrame({
        "pretty_formula": formula_comb,
        "spacegroup_relax": spg_comb,
        "enthalpy_formation_atom": energy_comb,
    })

    dataset = InferenceCrystalDataset(df, max_nsites=max_nsites, lattice_type=lattice_type,random_shuffle_coordinates=random_shuffle_coordinates)

    if return_df:
        return df, dataset

    return dataset
