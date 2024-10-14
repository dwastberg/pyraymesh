from dataclasses import dataclass, field
import numpy as np


@dataclass
class IntersectionResult:
    """
    A class to store the intersection results of a ray with a mesh.

    Attributes
    ----------
    coords : np.ndarray
        The coordinates of the intersection points.
    tri_ids : np.ndarray
        The triangle ids of the intersected triangles.
    distances : np.ndarray
        The distances from the ray origin to the intersection points.
    reflections : np.ndarray
        The reflection vectors at the intersection points.
    """

    coords: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    tri_ids: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    distances: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    reflections: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))

    def __len__(self) -> int:
        return len(self.coords)

    @property
    def hit_mask(self) -> np.ndarray:
        """
        Returns a boolean mask of hits
        """
        return ~np.isnan(self.coords[:, 0])

    @property
    def num_hits(self) -> int:
        """
        Returns the number of intersection points.
        :return: int
        """
        return (~np.isnan(self.coords[:, 0])).sum()
