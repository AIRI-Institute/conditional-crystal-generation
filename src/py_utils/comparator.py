import joblib
import numpy as np
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure


class PymatgenComparator:
    def __init__(
        self,
        params_list=[
            {"ltol": 0.2, "stol": 0.3, "angle_tol": 5},
            {"ltol": 0.4, "stol": 0.6, "angle_tol": 10},
            {"ltol": 1.0, "stol": 1.5, "angle_tol": 25},
        ],
        elm_str_path="../data/element.pkl",
    ) -> None:
        self.elm_str = np.array(joblib.load(elm_str_path))

        self.matchers = [StructureMatcher(**param) for param in params_list]

    def _form_up_structure(self, one_hot_vectors, coordinates_input, lattice):
        pred_elm = np.argmax(one_hot_vectors, axis=1)
        pred_elm = self.elm_str[pred_elm]

        struct = Structure(lattice=lattice, species=pred_elm, coords=coordinates_input)
        return struct

    def calculate_compares(
        self,
        element_matrix: torch.Tensor,
        n_sites: torch.Tensor,
        coordinates_truth: torch.Tensor,
        lattice_truth: torch.Tensor,
        coordinates_pred: torch.Tensor,
        lattice_pred: torch.Tensor,
    ) -> np.ndarray:
        """
        Returns boolean np.array of shape [matchers_count, batch_size]
        """
        element_matrix = element_matrix.cpu().detach().numpy()
        coordinates_truth = coordinates_truth.cpu().detach().numpy()
        lattice_truth = lattice_truth.cpu().detach().numpy()
        coordinates_pred = coordinates_pred.cpu().detach().numpy()
        lattice_pred = lattice_pred.cpu().detach().numpy()

        x1_structures = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(self._form_up_structure)(
                element_matrix[i, : n_sites[i]],
                coordinates_truth[i][: n_sites[i]],
                lattice_truth[i],
            )
            for i in range(len(element_matrix))
        )

        pred_structures = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(self._form_up_structure)(
                element_matrix[i, : n_sites[i]],
                coordinates_pred[i][: n_sites[i]],
                lattice_pred[i],
            )
            for i in range(len(element_matrix))
        )

        compares_batch = []

        for matcher in self.matchers:
            compares = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(matcher.fit)(pred, target)
                for pred, target in zip(pred_structures, x1_structures)
            )
            compares_batch.append(compares)

        return np.array(compares_batch)
