import pytest
import numpy as np

from pyraymesh.ray_functions import (
    cone_direction_vectors,
    sphere_direction_vectors,
    hammersley_sphere_direction_vectors,
)


def test_cone_direction_vectors():
    cone_angle = 45
    num_rays = 200
    cone_directions = cone_direction_vectors([0, 0, -1], cone_angle, num_rays)
    assert len(cone_directions) == num_rays
    vector_lengths = np.linalg.norm(cone_directions, axis=1)
    assert np.allclose(vector_lengths, 1)

    assert cone_directions[:, 2].min() == -1
    assert cone_directions[:, 2].max() < -0.5
    vector_lengths = np.linalg.norm(cone_directions, axis=1)
    assert np.allclose(vector_lengths, 1)


def test_sphere_direction_vectors():
    num_rays = 1000
    sphere_directions = sphere_direction_vectors(num_rays)
    assert len(sphere_directions) == num_rays
    vector_lengths = np.linalg.norm(sphere_directions, axis=1)
    assert np.allclose(vector_lengths, 1)

    assert sphere_directions[:, 0].max() > 0.995
    assert sphere_directions[:, 0].min() < -0.995

    assert sphere_directions[:, 1].max() > 0.995
    assert sphere_directions[:, 1].min() < -0.995

    assert sphere_directions[:, 2].max() > 0.995
    assert sphere_directions[:, 2].min() < -0.995


def test_hamersly_sphere_direction_vectors():
    num_rays = 1000
    sphere_directions = hammersley_sphere_direction_vectors(num_rays)
    assert len(sphere_directions) == num_rays
    vector_lengths = np.linalg.norm(sphere_directions, axis=1)
    assert np.allclose(vector_lengths, 1)

    assert sphere_directions[:, 0].max() > 0.995
    assert sphere_directions[:, 0].min() < -0.995

    assert sphere_directions[:, 1].max() > 0.995
    assert sphere_directions[:, 1].min() < -0.995

    assert sphere_directions[:, 2].max() > 0.995
    assert sphere_directions[:, 2].min() < -0.995
