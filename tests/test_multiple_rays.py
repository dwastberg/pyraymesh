import pytest

from pyraymesh import Mesh
import numpy as np


def mesh_plane():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [2, 3, 0]])
    return Mesh(vertices, faces)


def rays():
    ray_origin = np.array(
        [
            [0.01, 0.002, 1],
            [0.3, 0.4, 1],
            [0.5, 0.6, 1],
            [0.1, 0.2, 1],
            [0.3, 0.4, 1],
            [0.5, 0.6, 1],
        ]
    )
    ray_direction = np.array(
        [[0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
    )
    return ray_origin, ray_direction


def test_multiple_ray_intersect():
    m = mesh_plane()
    m.build("medium")
    ray_origin, ray_direction = rays()
    result = m.intersect(ray_origin, ray_direction)

    assert len(result) == 6
    assert (result.num_hits) == 3

    assert result.coords[0][0] == 0.01
    assert result.coords[0][1] == 0.002
    assert result.coords[0][2] == 0
    assert result.coords[1][0] == 0.3
    assert result.coords[1][1] == 0.4
    assert result.coords[1][2] == 0

    assert np.isnan(result.coords[5][0])
    assert np.isnan(result.coords[5][1])
    assert np.isnan(result.coords[5][2])

    assert len(result.tri_ids) == 6
    assert result.tri_ids[0] == 0
    assert result.tri_ids[1] == 1
    assert result.tri_ids[5] == -1

    assert len(result.distances) == 6
    assert result.distances[0] == 1

    hit_mask = result.hit_mask
    hit_coords = result.coords[hit_mask]
    assert len(hit_coords) == 3
    for h in hit_coords:
        assert np.isnan(h).any() == False


def test_multiple_ray_occlude():
    m = mesh_plane()
    m.build("medium")
    ray_origin, ray_direction = rays()
    result = m.occlusion(ray_origin, ray_direction)

    assert len(result) == 6
    assert result[0] == True
    assert result[1] == True
    assert result[2] == True
    assert result[3] == False
    assert result[4] == False
    assert result[5] == False
