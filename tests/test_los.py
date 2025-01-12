import pytest

from pyraymesh import Mesh

import numpy as np


def mesh_plane():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [2, 3, 0]])
    return Mesh(vertices, faces)


def mesh_cube():
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [2, 3, 0],
            [4, 5, 6],
            [6, 7, 4],
            [0, 4, 7],
            [7, 3, 0],
            [1, 5, 6],
            [6, 2, 1],
            [0, 1, 5],
            [5, 4, 0],
            [3, 7, 6],
            [6, 2, 3],
        ]
    )
    return Mesh(vertices, faces)


def test_no_los():
    m = mesh_plane()
    m.build("medium")
    origin_point = [0.5, 0.5, 1]
    target_point = [0.5, 0.5, -1]
    result = m.line_of_sight(origin_point, target_point)
    assert not result[0]


def test_los():
    m = mesh_plane()
    m.build("medium")
    origin_point = [0.5, 0.5, 1]
    target_point = [0.5, 0.5, 0.1]
    result = m.line_of_sight(origin_point, target_point)
    assert result[0]


def test_los_on_surface():
    m = mesh_plane()
    m.build("medium")
    origin_point = [0.5, 0.5, 1]
    target_point = [0.5, 0.5, 0]
    result = m.line_of_sight(origin_point, target_point)
    assert result[0]


def test_los_inside_geom():
    m = mesh_cube()
    m.build("medium")
    origin_point = [0.5, 0.5, 0.1]
    target_point = [0.5, 0.5, 0.8]
    result = m.line_of_sight(origin_point, target_point)
    assert result[0]


def test_multiple_origins():
    m = mesh_plane()
    m.build("medium")
    origin_point = [[0.5, 0.5, 1], [0.5, 0.5, 0.1], [0.5, 0.5, -0.1]]
    target_point = [0.5, 0.5, -1]
    result = m.line_of_sight(origin_point, target_point)
    assert np.all(result == [False, False, True])


def test_multiple_targets():
    m = mesh_cube()
    m.build("medium")
    origin_point = [0.5, 0.5, 20]
    target_point = [[0.5, 0.5, 1.1], [10, 10, 0], [0.5, 0.5, -1]]
    result = m.line_of_sight(origin_point, target_point)
    assert np.all(result == [True, True, False])
