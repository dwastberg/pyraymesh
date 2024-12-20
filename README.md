# pyraymesh

## Description

`pyraymesh` is a Python library for performing ray intersection and occlusion
tests on 3D meshes using a Bounding Volume Hierarchy (BVH). The library uses
the C++ library [bvh](https://github.com/madmann91/bvh) for building the BVH and performing the intersection tests.

While this library is reasonably fast for simpler meshes (benchmarks coming soon), it is not as fast as Embree, espcially for larger and more complex meshes. However, it does not
have any dependencies on external libraries, and is thus easier to install and use.

## Installation

Install the package either by 

```sh
pip install pyraymesh
```

or cloning the repo and using pip:

```sh   
pip install .
```

Note that the package requires a C++ compiler to build the C++ extension.

## Usage

### Building the BVH

To build the BVH for a mesh:

```python
from pyraymesh import Mesh
import numpy as np

vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
faces = np.array([[0, 1, 2], [2, 3, 0]])
mesh = Mesh(vertices, faces)
mesh.build("medium")
```

The `build` method takes a string argument that specifies the BVH build type, which can be one of the following:
"low", "medium" and "high". The build type determines the trade-off between build time and query time. For most cases
"medium" is almost always the right choice.

### Ray Intersection

To perform ray intersection tests:

```python
ray_origin = [[0.1, 0.2, 1], [0.2, 0.1, 1]]
ray_direction = [[0, 0, -1], [0, 0, 1]]
## or 
ray_origin = [[0.1, 0.2, 1], [0.2, 0.1, 1]]
ray_direction = [0, 0, -1]  # multiple rays with same direction
## or 
ray_origin = [0.1, 0.2, 1]
ray_direction = [[0, 0, -1], [0, 0, 1]]  # multiple rays with same origin

result = mesh.intersect(ray_origin, ray_direction, tnear=0, tfar=1000)
print(result.num_hits)
print(result.coords)
print(result.tri_ids)
print(result.distances)
```

`tnear` and `tfar` can be scalars or lists of the same length as the number of rays. If they are scalars, the same
value will be used for all rays. If they are lists, each value will be used for the corresponding ray.

If you set `tnear` to a value greater than 0, the intersection tests will ignore any intersections that are closer 
than `tnear`. Similarly, if you set `tfar` to a value less than infinity, the intersection tests will ignore any 
intersections that are farther than `tfar`.



### Reflections

If you want to get the reflection of the rays, add the `calculate_reflections = True` 
parameter to the `intersect` method:

```python
ray_origin = [[0.1, 0.2, 1], [0.2, 0.1, 1]]
ray_direction = [[0, 0, -1], [0, 0, 1]]
result = mesh.intersect(ray_origin, ray_direction, tnear=0, tfar=1000, calculate_reflections=True)
print(result.reflections)
```
results.reflections is a list  of noramlized vectors representing the directions of the 
reflection of the rays. Only do this if you need the reflections, as it will slow down the
intersection tests.



### Occlusion Test

If you just care about whether a ray is occluded or not (i.e., you don't care about
the intersection point) you can use the `occlusion` method which is faster than the
`intersect` method and just returns an array of booleans.

```python
ray_origin = [[0.1, 0.2, 1], [0.2, 0.1, 1]]
ray_direction = [[0, 0, -1], [0, 0, 1]]
occluded = mesh.occlusion(ray_origin, ray_direction)
print(occluded)
```

### Parallelization

The `intersect` and `occlusion` methods can be parallelized by passing `threads` parameter when calling the methods:

```python
result = mesh.intersect(ray_origin, ray_direction, tnear=0, tfar=1000, threads=4)
```

The `threads` parameter specifies the number of threads to use for the intersection tests. If set to `-1`, 
the number of threads will be equal to the number of cores on the machine. In general you shouldn't set the number of 
threads to be greater than the number of cores on the machine.

For a small number of rays, the overhead of parallelization might make the parallel version slower than the serial
version, so it is recommended to test the performance of both versions for your specific use case.


## Testing

To run the tests:

```sh
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
