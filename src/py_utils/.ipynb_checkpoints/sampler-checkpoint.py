from ast import List
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, Dataset
from src.py_utils.crystal_dataset import CrystalDataset


def filter_polymorphs(
    dataframe,
    min_polymorphs: int = 5,
    min_energy=-5,
    max_energy=5,
):
    """
    This function cleans a given dataframe from duplicates and polymorph groups that have less than 'min_polymorphs' number of structures
    """
    dataframe["index"] = dataframe.index

    dataframe = dataframe[dataframe["enthalpy_formation_atom"] < max_energy]
    dataframe = dataframe[dataframe["enthalpy_formation_atom"] > min_energy]

    formula_aggs = dataframe.groupby(by=["pretty_formula"])[["index"]].agg(list)
    polymorph_counts = (
        pd.DataFrame(formula_aggs["index"].apply(len))
        .rename(columns={"index": "num_polymorphs"})
        .reset_index(drop=False)
    )

    dataframe = dataframe.merge(polymorph_counts, on="pretty_formula")
    return (
        dataframe[dataframe["num_polymorphs"] >= min_polymorphs]
        .drop(columns=["num_polymorphs", "index"])
        .reset_index(drop=True)
    )


def pairwise_combs_diff(*arrays):
    """
    This function takes numerous arrays and returns cartesian product of given arrays
    In fact, our task uses two arrays of indexes.
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def pairwise_combs_same(array):
    """
    This function returns cartesian product arrays: (array, array) with unique combinations.
    For example there is no combination of two exact elements of 'array'
    """
    array = np.array(array)
    n = len(array)
    L = n * (n - 1) // 2
    out = np.empty((L, 2), dtype=array.dtype)
    m = ~np.tri(len(array), dtype=bool)
    out[:, 0] = np.broadcast_to(array[:, None], (n, n))[m]
    out[:, 1] = np.broadcast_to(array, (n, n))[m]
    return out


def get_all_pairs(
    bad: np.array,
    good: np.array,
):
    """
    This function takes indexes of bad structures and indexes of good structures and returns their combinations.
    Good structures are two the structure that have the least formation energy in their polymorph group.
    """
    good_pairs = np.vstack((pairwise_combs_diff(bad, good), pairwise_combs_same(good)))
    bad_pairs = pairwise_combs_same(bad)
    return bad_pairs, good_pairs


class PairDataset:
    """
    This class implements a dataset, that is capable of iterating over pairs of structures.
    This class can be used as a test or train dataset.
    Test dataset is as usual dataset, but train dataset updates pairs when iteration loop ends.
    """

    def __init__(
        self,
        crystal_dataset: CrystalDataset,
        num_pairs,
        test_pairs=None,
        bad_pairs_array=None,
        good_pairs_array=None,
        sampling_strategy="train_good",
        bad_pairs_choice_probs=None,
        good_pairs_choice_probs=None,
    ):
        if test_pairs is None and bad_pairs_array is None and good_pairs_array is None:
            raise Exception(
                "At least one of the fields: 'test_pairs', 'bad_pairs_array', 'good_pairs_array' should be not None"
            )
        elif test_pairs is not None and (
            bad_pairs_array is not None or good_pairs_array is not None
        ):
            raise Exception(
                "If 'test_pairs' is not None, then 'bad_pairs_array', 'good_pairs_array' should be None"
            )
        elif (bad_pairs_array is not None and good_pairs_array is None) or (
            bad_pairs_array is None and good_pairs_array is not None
        ):
            raise Exception(
                "If 'bad_pairs_array' is not None, then 'good_pairs_array' should be not None too"
            )
        assert sampling_strategy in [
            "train_good",
            "train_bad",
            "train_good_nsites_balaned",
        ]
        self.sampling_strategy = sampling_strategy
        self.dataset = crystal_dataset
        self.num_pairs = num_pairs
        self.bad_pairs_array = bad_pairs_array
        self.good_pairs_array = good_pairs_array

        self.bad_pairs_choice_probs = bad_pairs_choice_probs
        self.good_pairs_choice_probs = good_pairs_choice_probs
        if bad_pairs_array is not None:
            self.pairs = self.get_train_pairs()
        else:
            self.pairs = test_pairs

    def __len__(
        self,
    ):
        return len(self.pairs)

    def get_train_pairs(self):
        if self.sampling_strategy == "train_good" or "train_good_nsites_balaned":
            indexes_bad = np.random.choice(
                np.arange(self.bad_pairs_array.shape[0]),
                min(self.num_pairs // 2, self.bad_pairs_array.shape[0]),
                replace=False,
                p=self.bad_pairs_choice_probs,
            )
            indexes_good = np.random.choice(
                np.arange(self.good_pairs_array.shape[0]),
                min(self.num_pairs // 2, self.good_pairs_array.shape[0]),
                replace=False,
                p=self.good_pairs_choice_probs,
            )
            return np.vstack(
                (
                    self.bad_pairs_array[indexes_bad],
                    self.good_pairs_array[indexes_good],
                )
            )
        elif self.sampling_strategy == "train_bad":
            indexes_bad = np.random.choice(
                np.arange(self.bad_pairs_array.shape[0]),
                min(int(self.num_pairs), self.bad_pairs_array.shape[0]),
                replace=False,
                p=self.bad_pairs_choice_probs,
            )
            return self.bad_pairs_array[indexes_bad]

    def __getitem__(self, index):
        x0_index, x1_index = self.pairs[index]

        x0 = self.dataset[x0_index]
        x1 = self.dataset[x1_index]

        result = {
            "x0_coordinates_with_lattice": x0["coordinates_with_lattice"],
            "x1_coordinates_with_lattice": x1["coordinates_with_lattice"],
            "element_matrix": x0["element_matrix"],
            "elemental_property_matrix": x0["elemental_property_matrix"],
            "spg": x1["spg"],
            "energy": x1["energy"] - x0["energy"],
            "nsites": x0["n_sites"],
            "x0_spacegroup_number": x0["spacegroup_number"],
            "x1_spacegroup_number": x1["spacegroup_number"],
            "x0_energy": x0["energy"],
            "x1_energy": x1["energy"],
            "formula": x0["formula"],
        }

        if index + 1 == len(self) and self.bad_pairs_array is not None:
            self.pairs = self.get_train_pairs()
        elif index >= len(self):
            IndexError()
        return result


def sort_pair(pairs, energy_array, disable_tqdm=True):
    for index, pair in tqdm(
        enumerate(pairs), desc="Sorting pairs of structures", disable=disable_tqdm
    ):
        if energy_array[pair[0]] < energy_array[pair[1]]:
            pairs[index][0], pairs[index][1] = pairs[index][1], pairs[index][0]
    return pairs


def prepare_dataframe(
    dataframe,
    min_polymorphs=5,
):
    """
    This function takes a dataframe and other parameters as an input.
    It cleans from duplicates and small polymorph groups, creates and returns CrystalDataset
    """
    if type(dataframe) is not pd.DataFrame:
        dataframe = pd.read_csv(dataframe)

    dataframe = (
        dataframe.drop_duplicates("cif")
        .sort_values("enthalpy_formation_atom", ascending=False)
        .reset_index(drop=True)
    )
    print(f"shape before processing: {dataframe.shape}")

    dataframe = filter_polymorphs(dataframe, min_polymorphs=min_polymorphs)
    print(f"shape after processing: {dataframe.shape}")

    return dataframe


def get_dataloaders_pairs(
    dataframe_path,
    max_nsites,
    min_polymorphs=5,
    lattice_type="3_by_3",
    max_pairs_per_group=10,
    test_size=0.2,
    batch_size=64,
    sampling_strategy="train_good",
    random_state=69,
    num_workers=0,
):
    """
    This function takes CrystalDataset and other parameters as input, initializes test and train datasets and dataloaders
    and resutns them
    """
    dataframe = prepare_dataframe(dataframe_path, min_polymorphs=min_polymorphs)

    dataframe["index"] = dataframe.index

    poly_indexes = dataframe.groupby(by=["pretty_formula"])[["index"]].agg(list)

    bad_pairs_array = []
    good_pairs_array = []
    for poly_group_indexes in tqdm(
        poly_indexes["index"].values, desc="Making pairs for every poly group"
    ):
        bad_structures = poly_group_indexes[:-1]
        good_structures = poly_group_indexes[-1:]
        bad_pairs, good_pairs = get_all_pairs(
            np.array(bad_structures),
            np.array(good_structures),
        )
        bad_pairs_array.extend(bad_pairs)
        good_pairs_array.extend(good_pairs)

    bad_pairs_array = sort_pair(
        bad_pairs_array, dataframe["enthalpy_formation_atom"].values
    )
    good_pairs_array = sort_pair(
        good_pairs_array, dataframe["enthalpy_formation_atom"].values
    )
    # calculating the amount of train and test samples
    test_counts = int(max_pairs_per_group * test_size * len(poly_indexes))
    train_counts = int(max_pairs_per_group * (1 - test_size) * len(poly_indexes))

    # test pairs definement.
    np.random.seed(random_state)
    test_indexes = np.random.choice(
        np.arange(len(good_pairs_array)), test_counts, replace=False
    )
    good_pairs_array = np.array(good_pairs_array)
    bad_pairs_array = np.array(bad_pairs_array)

    test_pairs = good_pairs_array[test_indexes]
    good_pairs_array = good_pairs_array[
        np.setdiff1d(np.arange(len(good_pairs_array)), test_indexes).astype(int)
    ]

    dataset = CrystalDataset(
        dataframe=dataframe,
        max_nsites=max_nsites,
        lattice_type=lattice_type,
    )

    test_dataset = PairDataset(
        dataset,
        test_counts,
        test_pairs=test_pairs,
        bad_pairs_array=None,
        good_pairs_array=None,
        sampling_strategy=sampling_strategy,
    )
    train_dataset = PairDataset(
        dataset,
        train_counts,
        test_pairs=None,
        bad_pairs_array=bad_pairs_array,
        good_pairs_array=good_pairs_array,
        sampling_strategy=sampling_strategy,
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return test_dataset, train_dataset, test_dataloader, train_dataloader


def get_balanced_dataloaders_pairs(
    base_dataframe: pd.DataFrame,
    train_formulas,
    test_formulas,
    max_nsites=64,
    min_polymorphs=5,
    lattice_type="3_by_3",
    avg_pairs_per_group=(5, 1),  # tuple - (train_pairs, test_pairs)
    batch_size=256,
    sampling_strategy="train_good_nsites_balaned",  # train_good
    random_state=69,
    num_workers=0,
    disable_tqdm=True,
    top_k_good=1,
    apply_energy_noising=False,
):
    """
    This function takes dataframe and stratifies it using balanced technique
    """
    base_dataframe = prepare_dataframe(base_dataframe, min_polymorphs=min_polymorphs)

    train_df = base_dataframe[
        base_dataframe["pretty_formula"].isin(train_formulas)
    ].reset_index(drop=True)
    test_df = base_dataframe[
        base_dataframe["pretty_formula"].isin(test_formulas)
    ].reset_index(drop=True)

    for dataframe, df_type, avg_pairs in zip(
        [train_df, test_df], ["train", "test"], avg_pairs_per_group
    ):
        dataframe["index"] = dataframe.index

        poly_indexes = dataframe.groupby(by=["pretty_formula"])[["index"]].agg(list)

        bad_pairs_array = []
        good_pairs_array = []
        for poly_group_indexes in tqdm(
            poly_indexes["index"].values,
            desc="Making pairs for every poly group",
            disable=disable_tqdm,
        ):
            bad_structures = poly_group_indexes[:-top_k_good]
            good_structures = poly_group_indexes[-top_k_good:]
            bad_pairs, good_pairs = get_all_pairs(
                np.array(bad_structures),
                np.array(good_structures),
            )
            bad_pairs_array.extend(bad_pairs)
            good_pairs_array.extend(good_pairs)

        bad_pairs_array = np.array(
            sort_pair(
                bad_pairs_array,
                dataframe["enthalpy_formation_atom"].values,
                disable_tqdm=disable_tqdm,
            )
        )
        good_pairs_array = np.array(
            sort_pair(
                good_pairs_array,
                dataframe["enthalpy_formation_atom"].values,
                disable_tqdm=disable_tqdm,
            )
        )

        pairs_count = int(avg_pairs * len(poly_indexes))

        def get_probabilities(pairs_nsites):
            pairs_nsites_unique, pairs_nsites_counts = np.unique(
                pairs_nsites, return_counts=True
            )
            probs = 1 / pairs_nsites_counts

            b = np.zeros_like(pairs_nsites)
            for idx, u in enumerate(pairs_nsites_unique):
                b += idx * (pairs_nsites == u)
            return probs[b] / probs[b].sum()

        if sampling_strategy == "train_good_nsites_balaned":
            bad_pairs_nsites = dataframe["nsites"].to_numpy()[bad_pairs_array[:, 0]]
            good_pairs_nsites = dataframe["nsites"].to_numpy()[good_pairs_array[:, 0]]
            bad_pairs_probs = get_probabilities(bad_pairs_nsites)
            good_pairs_probs = get_probabilities(good_pairs_nsites)
        else:
            bad_pairs_probs = None
            good_pairs_probs = None

        dataset = CrystalDataset(
            dataframe=dataframe,
            max_nsites=max_nsites,
            lattice_type=lattice_type,
            apply_energy_noising=apply_energy_noising if df_type == "train" else False,
        )

        if df_type == "train":

            train_dataset = PairDataset(
                dataset,
                pairs_count,
                test_pairs=None,
                bad_pairs_array=bad_pairs_array,
                good_pairs_array=good_pairs_array,
                sampling_strategy=sampling_strategy,
                good_pairs_choice_probs=good_pairs_probs,
                bad_pairs_choice_probs=bad_pairs_probs,
            )
        else:
            np.random.seed(random_state)

            good_pairs_indexes = np.random.choice(
                np.arange(len(good_pairs_array)),
                pairs_count,
                replace=False,
                p=good_pairs_probs,
            )
            test_pairs = good_pairs_array[good_pairs_indexes]

            test_dataset = PairDataset(
                dataset,
                pairs_count,
                test_pairs=test_pairs,
                bad_pairs_array=None,
                good_pairs_array=None,
                sampling_strategy=sampling_strategy,
            )

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return test_dataset, train_dataset, test_dataloader, train_dataloader


def get_balanced_dataloaders_non_pairs(
    base_dataframe: pd.DataFrame,
    train_formulas,
    test_formulas,
    max_nsites=64,
    min_polymorphs=5,
    lattice_type="3_by_3",
    avg_structures_per_group=10,
    batch_size=256,
    sampling_strategy="train_good_nsites_balaned",  # train_good
    random_state=69,
    num_workers=0,
    disable_tqdm=True,
    top_k_good=1,
    apply_energy_noising=False,
    train_type="train_full",  # or train_balanced
):
    """
    This function takes dataframe and stratifies it using balanced technique
    """
    base_dataframe = prepare_dataframe(base_dataframe, min_polymorphs=min_polymorphs)

    train_df = base_dataframe[
        base_dataframe["pretty_formula"].isin(train_formulas)
    ].reset_index(drop=True)
    test_df = base_dataframe[
        base_dataframe["pretty_formula"].isin(test_formulas)
    ].reset_index(drop=True)

    for dataframe, df_type in zip([train_df, test_df], [train_type, "test"]):
        if df_type == "train_full":
            train_dataset = CrystalDataset(
                dataframe=dataframe,
                max_nsites=max_nsites,
                lattice_type=lattice_type,
                apply_energy_noising=apply_energy_noising,
            )
            continue

        dataframe["index"] = dataframe.index

        poly_indexes = dataframe.groupby(by=["pretty_formula"])[["index"]].agg(list)

        bad_structures_array = []
        good_structures_array = []
        for poly_group_indexes in tqdm(
            poly_indexes["index"].values,
            desc="Making pairs for every poly group",
            disable=disable_tqdm,
        ):
            bad_structures = poly_group_indexes[:-top_k_good]
            good_structures = poly_group_indexes[-top_k_good:]
            bad_structures_array.extend(bad_structures)
            good_structures_array.extend(good_structures)

        bad_structures_array = np.array(bad_structures_array)
        good_structures_array = np.array(good_structures_array)

        def get_probabilities(pairs_nsites):
            pairs_nsites_unique, pairs_nsites_counts = np.unique(
                pairs_nsites, return_counts=True
            )
            probs = 1 / pairs_nsites_counts

            b = np.zeros_like(pairs_nsites)
            for idx, u in enumerate(pairs_nsites_unique):
                b += idx * (pairs_nsites == u)

            return probs[b] / probs[b].sum()

        if sampling_strategy == "train_good_nsites_balaned":
            bad_nsites = dataframe["nsites"].to_numpy()[bad_structures_array]
            good_nsites = dataframe["nsites"].to_numpy()[good_structures_array]
            bad_structs_probs = get_probabilities(bad_nsites)
            good_structs_probs = get_probabilities(good_nsites)
        else:
            bad_structs_probs = None
            good_structs_probs = None

        total_structures = int(avg_structures_per_group * len(poly_indexes))

        if df_type == "train_balanced":
            np.random.seed(random_state)

            good_indexes = np.random.choice(
                np.arange(len(good_structures_array)),
                int(total_structures * 0.5),
                replace=False,
                p=good_structs_probs,
            )
            bad_indexes = np.random.choice(
                np.arange(len(bad_structures_array)),
                int(total_structures * 0.5),
                replace=False,
                p=bad_structs_probs,
            )
            train_ids = np.concatenate(
                (good_structures_array[good_indexes], bad_structures_array[bad_indexes])
            )

            train_dataset = CrystalDataset(
                dataframe=dataframe.loc[train_ids],
                max_nsites=max_nsites,
                lattice_type=lattice_type,
                apply_energy_noising=apply_energy_noising,
            )

        elif df_type == "test":
            np.random.seed(random_state)

            good_indexes = np.random.choice(
                np.arange(len(good_structures_array)),
                total_structures,
                replace=False,
                p=good_structs_probs,
            )
            test_ids = good_structures_array[good_indexes]

            test_dataset = CrystalDataset(
                dataframe=dataframe.loc[test_ids],
                max_nsites=max_nsites,
                lattice_type=lattice_type,
                apply_energy_noising=apply_energy_noising,
            )

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return test_dataset, train_dataset, test_dataloader, train_dataloader
