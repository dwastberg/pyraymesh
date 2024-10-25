import numpy as np

from . import _bvh_bind_ext
from .IntersectionResult import IntersectionResult
from typing import List, Iterable, Union
import os
import numbers



def _prep_rays(ray_origin, ray_direction, tnear, tfar):
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
    if isinstance(tnear, Iterable):
        tnear = np.array(tnear, dtype=np.float32)
    if isinstance(tfar, Iterable):
        tfar = np.array(tfar, dtype=np.float32)
    if isinstance(tnear, numbers.Number):
        tnear = np.full(len(ray_origin), tnear, dtype=np.float32)
    if isinstance(tfar, numbers.Number):
        tfar = np.full(len(ray_origin), tfar, dtype=np.float32)
    if len(tnear) != len(ray_origin):
        raise ValueError(
            "tnear must have the same length as ray_origin or be a single value"
        )
    if len(tfar) != len(ray_origin):
        raise ValueError(
            "tfar must have the same length as ray_origin or be a single value"
        )
    return ray_origin, ray_direction, tnear, tfar


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

    def build(self, quality: str = "medium") -> None:
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
        ray_origin: Iterable[Iterable[float]],
        ray_direction: Iterable[Iterable[float]],
        tnear: Union[float,Iterable[float]] = 0,
        tfar: Union[float,Iterable[float]]  = np.finfo(np.float32).max,
        calculate_reflections: bool = False,
        threads: int = 1,
    ) -> IntersectionResult:
        """
        Intersects the rays with the mesh.

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
        ray_origin, ray_direction, tnear, tfar = _prep_rays(ray_origin, ray_direction, tnear, tfar)

        if threads < 1:
            threads = os.cpu_count()

        if calculate_reflections:
            coords, tri_ids, distances, reflections = _bvh_bind_ext.intersect_bvh(
                self._bvh, ray_origin, ray_direction, tnear, tfar, calculate_reflections, self.robust, threads
            )
        else:
            coords, tri_ids, distances = _bvh_bind_ext.intersect_bvh(
                self._bvh, ray_origin, ray_direction, tnear, tfar, calculate_reflections, self.robust, threads
            )
            reflections = np.empty((0, 3))

        return IntersectionResult(coords=coords, tri_ids=tri_ids, distances=distances, reflections=reflections)

    def occlusion(
        self,
        ray_origin: Iterable[float],
        ray_direction: Iterable[float],
        tnear=0,
        tfar=np.finfo(np.float32).max,
        threads: int = 1,
    ) -> np.ndarray:
        """
        Checks for occlusion along the rays with the BVH (Bounding Volume Hierarchy) of the mesh.

        Parameters:
        ray_origin (array-like): The origin points of the rays.
        ray_direction (array-like): The direction vectors of the rays.
        tnear (float, optional): The minimum distance along the ray to consider for occlusion. Defaults to 0.
        tfar (float, optional): The maximum distance along the ray to consider for occlusion. Defaults to np.inf.

        Returns:
        bool: (list of bool) A list of boolean values indicating whether the ray is occluded.

        Raises:
        ValueError: If the BVH is not built and cannot be built with the specified quality.
        """
        if not self.is_built:
            print("BVH not built, building now with medium quality")
            self.build("medium")
        if threads < 1:
            threads = os.cpu_count()
        ray_origin, ray_direction, tnear,tfar = _prep_rays(ray_origin, ray_direction, tnear, tfar)
        result = _bvh_bind_ext.occlude_bvh(
            self._bvh, ray_origin, ray_direction, tnear, tfar, self.robust, threads
        )

        return result

