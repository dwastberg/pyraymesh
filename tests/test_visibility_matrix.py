import pytest

from pyraymesh import Mesh

import numpy as np


def mesh_plane():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [2, 3, 0]])
    return Mesh(vertices, faces)


def view_points():
    return np.array([[0.5, 0.5, 1], [0.5, 0.5, 0.1], [0.5, 0.5, -1], [0.5, 0.5, -1]])


def test_visibility_matrix():
    m = mesh_plane()
    m.build("medium")
    pts = view_points()

    result = m.visibility_matrix(pts)
    expected = np.array(
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, True],
            [False, False, True, True],
        ]
    )
    assert np.all(result == expected)
