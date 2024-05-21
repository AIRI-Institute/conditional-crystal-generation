import re
from math import gcd
import pandas as pd
import sys

from tqdm import tqdm

sys.path.append("../")

from  src.py_utils.skmultilearn_iterative_split import IterativeStratification
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter


def simplify_formula(formula):
    reg = re.findall("[A-Z][a-z]?|[0-9]+", formula)
    chem_elements, numbers = reg[::2], reg[1::2]
    numbers = list(map(int, numbers))
    gcd_num = gcd(*numbers)
    numbers = [i // gcd_num for i in numbers]

    simplified_formula = "".join(
        f"{chem_el}{n}" for chem_el, n in zip(chem_elements, numbers)
    )
    return simplified_formula


def train_test_split_with_chemical_balance(
    df: pd.DataFrame, test_size=0.2, random_state=42, verbose=False
):
    """
    Splits a DataFrame into training and testing sets while considering chemical balance.

    Parameters:
    - df (pd.DataFrame): The original DataFrame to split. Must contain 'elements' and 'pretty_formula' columns
    - test_size (float): The proportion of the dataset to include in the test split. Default is 0.2.
    - random_state (int): Seed for random number generation. Default is 42.
    - verbose (bool): Flag to display information about data sizes. Default is False.

    Returns:
    - tuple: A tuple containing unique formulas of the training and testing sets.

    Structures with a single chemical formula are placed entirely in either the train or test set to prevent data leakage.
    This function divides the data while considering the chemical balance, maintaining proportions of elements.
    """

    np.random.seed(random_state)

    df = df.copy()
    df["elements"] = df["elements"].apply(eval).apply(sorted).apply(tuple)
    df["primal_formula"] = df["pretty_formula"].apply(simplify_formula)
    df["primal_formula"].unique().shape

    formulas_df = df[["primal_formula", "elements"]].drop_duplicates()
    unique_combinations = formulas_df["elements"].unique()
    np.random.shuffle(unique_combinations)

    train_formulas, test_formulas = [], []

    for _, comb_df in formulas_df.groupby("elements"):
        if len(comb_df) >= 2:
            train, test = train_test_split(
                comb_df["primal_formula"],
                test_size=max(1, int(test_size * len(comb_df))),
                random_state=random_state,
            )
            train_formulas.extend(train)
            test_formulas.extend(test)

    solo_df = formulas_df.groupby("elements").filter(lambda group: len(group) == 1)
    solo_df_onehot = solo_df["elements"].str.join("|").str.get_dummies().astype(bool)

    stratifier = IterativeStratification(
        n_splits=2, order=2, sample_distribution_per_fold=[test_size, 1.0 - test_size]
    )
    train_indexes, test_indexes = next(
        stratifier.split(list(range(len(solo_df_onehot))), solo_df_onehot.astype(int))
    )

    train_solo, test_solo = (
        solo_df["primal_formula"].iloc[train_indexes],
        solo_df["primal_formula"].iloc[test_indexes],
    )

    train_df = df[
        (df["primal_formula"].isin(train_formulas))
        | (df["primal_formula"].isin(train_solo))
    ]
    test_df = df[
        (df["primal_formula"].isin(test_formulas))
        | (df["primal_formula"].isin(test_solo))
    ]

    assert train_df.shape[0] + test_df.shape[0] == df.shape[0]
    assert (
        len(
            set(train_df["pretty_formula"]).intersection(set(test_df["pretty_formula"]))
        )
        == 0
    )

    train_elements = train_df["elements"]
    test_elements = test_df["elements"]

    train_elements = list(item for t in train_elements for item in t)
    test_elements = list(item for t in test_elements for item in t)

    assert set(train_elements) == set(
        test_elements
    ), f"{set(train_elements) - set(test_elements)} {set(test_elements) - set(train_elements)}"

    train_counts = Counter(train_elements)
    test_counts = Counter(test_elements)

    for k in train_counts.keys():
        train_counts[k] /= len(train_elements)

    for k in test_counts.keys():
        test_counts[k] /= len(test_elements)
    deltas = {}

    for k in test_counts.keys():
        delta = train_counts[k] - test_counts[k]
        deltas[k] = delta

    diff = sum(abs(i) for i in deltas.values())

    if verbose:
        print(
            f"Train/test structures df size ratio : {train_df.shape[0] / test_df.shape[0] :.5f}"
        )  # Should be around 4 for 0.2 test_size
        print(f"Elements absolute difference: {diff :.5f}")  # Should be around 0.15

    return train_df["pretty_formula"].unique(), test_df["pretty_formula"].unique()
