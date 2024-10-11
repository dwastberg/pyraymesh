import numpy as np

from . import _bvh_bind_ext
from .IntersectionResult import IntersectionResult
from typing import List, Iterable, Union


def prep_rays(ray_origin, ray_direction, tmin=0, tfar=np.inf):
    ray_origin = np.array(ray_origin, dtype=np.float32)
    ray_direction = np.array(ray_direction, dtype=np.float32)
    if len(ray_origin.shape) == 1:
        if len(ray_origin) != 3:
            raise ValueError("ray_origin must have 3 elements")
        ray_origin = ray_origin[np.newaxis, :]
    if len(ray_direction.shape) == 1:
        if len(ray_direction) != 3:
            raise ValueError("ray_direction must have 3 elements")
        ray_direction = ray_direction[np.newaxis, :]
    if len(ray_origin) == 1 and len(ray_direction) > 1:
        ray_origin = np.repeat(ray_origin, len(ray_direction), axis=0)
    if len(ray_direction) == 1 and len(ray_origin) > 1:
        ray_direction = np.repeat(ray_direction, len(ray_origin), axis=0)
    if not len(ray_origin) == len(ray_direction):
        raise ValueError(
            "ray_origin and ray_direction must have the same length or one of them must have length 1"
        )
    return ray_origin, ray_direction


class Mesh:
    def __init__(
        self, vertices: Iterable[float], faces: Union[Iterable[int], None] = None
    ):
        """
        Initializes the Mesh object with vertices and optional faces.

        Parameters:
        vertices (array-like): An array of vertex coordinates.
        faces (array-like, optional): An array of face indices. Defaults to None. If None, the vertices are assumed to be a list of triangles.
        """
        self.vertices = vertices
        self.faces = faces
        self._normalize_mesh_data()
        self._bvh = None
        self.robust = True

    def _normalize_mesh_data(self):
        self.vertices = np.array(self.vertices, dtype=np.float32)
        if self.faces is not None:
            self.faces = np.array(self.faces, dtype=np.int32)
        if len(self.vertices.shape) == 1:
            self.vertices = self.vertices.reshape(-1, 3)
        if self.faces is not None and len(self.faces.shape) == 1:
            self.faces = self.faces.reshape(-1, 3)
        if self.faces is None:
            self.faces = np.arange(self.vertices.shape[0], dtype=np.int32).reshape(
                -1, 3
            )

    @property
    def is_built(self) -> bool:
        return self._bvh is not None

    def build(self, quality: str = "medium"):
        """
        Builds the BVH (Bounding Volume Hierarchy) for the mesh with the specified quality.

        Parameters:
        quality (str): The quality level for building the BVH. Must be one of 'low', 'medium', or 'high'.
                       Defaults to 'medium'.

        Raises:
        ValueError: If the quality is not one of 'low', 'medium', or 'high'.
        """
        quality = quality.lower()
        if quality not in ["low", "medium", "high"]:
            raise ValueError("Quality must be one of 'low', 'medium' or 'high'")
        if len(self.vertices) == 0 or len(self.faces) == 0:
            raise ValueError("Mesh is empty")
        self._bvh = _bvh_bind_ext.build_bvh(self.vertices, self.faces, quality)

    def intersect(
        self,
        ray_origin: Iterable[float],
        ray_direction: Iterable[float],
        tnear: float = 0,
        tfar: float = np.inf,
    ) -> IntersectionResult:
        """
        Intersects the rays with the BVH (Bounding Volume Hierarchy) of the mesh.

        Parameters:
        ray_origin (array-like): The origin points of the rays.
        ray_direction (array-like): The direction vectors of the rays.
        tnear (float, optional): The minimum distance along the ray to consider for intersections. Defaults to 0.
        tfar (float, optional): The maximum distance along the ray to consider for intersections. Defaults to np.inf.

        Returns:
        Hits: An object containing the intersection coordinates, triangle IDs, and distances.

        Raises:
        ValueError: If the BVH is not built and cannot be built with the specified quality.
        """
        if not self.is_built:
            print("BVH not built, building now with medium quality")
            self.build("medium")
            if not self.is_built:
                raise ValueError("failed to build BVH")
        ray_origin, ray_direction = prep_rays(ray_origin, ray_direction, tnear, tfar)

        coords, tri_ids, distances = _bvh_bind_ext.intersect_bvh(
            self._bvh, ray_origin, ray_direction, tnear, tfar, self.robust
        )

        return IntersectionResult(coords, tri_ids, distances)

    def occlusion(
        self,
        ray_origin: Iterable[float],
        ray_direction: Iterable[float],
        tmin=0,
        tfar=np.inf,
    ) -> np.ndarray:
        """
        Checks for occlusion along the rays with the BVH (Bounding Volume Hierarchy) of the mesh.

        Parameters:
        ray_origin (array-like): The origin points of the rays.
        ray_direction (array-like): The direction vectors of the rays.
        tmin (float, optional): The minimum distance along the ray to consider for occlusion. Defaults to 0.
        tfar (float, optional): The maximum distance along the ray to consider for occlusion. Defaults to np.inf.

        Returns:
        hist: (list of bool) A list of boolean values indicating whether the ray is occluded.

        Raises:
        ValueError: If the BVH is not built and cannot be built with the specified quality.
        """
        if not self.is_built:
            print("BVH not built, building now with medium quality")
            self.build("medium")
        ray_origin, ray_direction = prep_rays(ray_origin, ray_direction, tmin, tfar)

        return np.array(
            _bvh_bind_ext.occlude_bvh(
                self._bvh, ray_origin, ray_direction, tmin, tfar, self.robust
            )
        )
