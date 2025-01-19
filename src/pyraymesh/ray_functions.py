import numpy as np
from typing import Iterable, Tuple


def fibonacci_sphere_direction_vectors(n: int) -> np.ndarray:
    """
    Generate `n` points on a sphere using the Fibonacci spiral.

    Args:
        n (int): The number of points to generate.

    Returns:
        np.ndarray: An array of shape (n, 3) containing (x, y, z) coordinates of the points on the sphere.
    """
    # Create index array from 0 to n-1
    i = np.arange(n, dtype=float)

    # Calculate z coordinates using vectorized operation
    z = 1 - (2 * i + 1) / n

    # Calculate radius at each z
    r = np.sqrt(1 - z * z)

    # Calculate phi angle (golden ratio spiral)
    phi = i * (2 * np.pi / ((1 + np.sqrt(5)) / 2))

    # Convert to Cartesian coordinates in one step
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def hammersley_sphere_direction_vectors(n: int) -> np.ndarray:
    """
    Generate `n` points on a sphere using the Hammersley sequence.

    Args:
        n (int): The number of points to generate.

    Returns:
        np.ndarray: An array of shape (n, 3) containing (x, y, z) coordinates of the points on the sphere.
    """
    points = np.zeros((n, 3))

    def radical_inverse_2(n: int) -> float:
        result = 0
        factor = 1.0
        while n:
            factor *= 0.5
            result += factor * (n & 1)
            n >>= 1
        return result

    for i in range(n):
        u = i / n
        v = radical_inverse_2(i)

        phi = 2 * np.pi * u
        cos_theta = 1 - 2 * v
        sin_theta = np.sqrt(1 - cos_theta * cos_theta)

        points[i] = [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta]

    return points


def cone_direction_vectors(
    direction: Iterable[float], angle_degrees: float, n: int
) -> np.ndarray:
    """
    Generate N direction vectors in a cone around a direction vector.

    Args:
       direction: (3,) array, central direction of cone
       angle_degrees: float, half-angle of cone in degrees
       n: int, number of rays to generate

    Returns:
       Array of shape (n,3) containing normalized direction vectors
    """
    # Normalize direction
    direction = direction / np.linalg.norm(direction)
    angle_rad = np.deg2rad(angle_degrees)
    cos_angle = np.cos(angle_rad)

    if abs(direction[0]) < abs(direction[1]):
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = np.array([0.0, 1.0, 0.0])

    # Create orthonormal basis
    u = np.cross(direction, u)
    u = u / np.linalg.norm(u)
    v = np.cross(direction, u)

    # Generate spiral on unit disk
    i = np.arange(n, dtype=float)
    z = 1.0 - (1.0 - cos_angle) * i / (n - 1)  # Map to [cos_angle, 1]
    r = np.sqrt(np.maximum(0, 1 - z * z))

    # spiral angles
    phi = i * 2 * np.pi / ((1 + np.sqrt(5)) / 2)

    x = r * np.cos(phi)
    y = r * np.sin(phi)
    local_dirs = np.column_stack([x, y, z])

    # Transform all directions to world space at once
    transform = np.column_stack([u, v, direction])
    world_dirs = local_dirs @ transform.T

    return world_dirs / np.linalg.norm(world_dirs, axis=1)[:, np.newaxis]


# use fibonacci sphere as default
sphere_direction_vectors = fibonacci_sphere_direction_vectors
