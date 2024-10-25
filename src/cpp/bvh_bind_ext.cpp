#include "types.h"
#include "utils.h"
#include "Accel.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/node.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/tri.h>
#include <iostream>

namespace nb = nanobind;
using namespace nb::literals;

constexpr Scalar ScalarNAN = std::numeric_limits<Scalar>::quiet_NaN();
constexpr size_t INVALID_ID = std::numeric_limits<size_t>::max();


auto build_bvh(const nb::ndarray<Scalar, nb::shape<-1, 3>> &vertices, const nb::ndarray<int, nb::shape<-1, 3>> &indices,
               const std::string &quality = "medium")
{
    std::vector<Tri> tris;
    tris.reserve(indices.shape(0));
    for (size_t i = 0; i < indices.shape(0); i++)
    {
        tris.emplace_back(
            Vec3(vertices(indices(i, 0), 0), vertices(indices(i, 0), 1), vertices(indices(i, 0), 2)),
            Vec3(vertices(indices(i, 1), 0), vertices(indices(i, 1), 1), vertices(indices(i, 1), 2)),
            Vec3(vertices(indices(i, 2), 0), vertices(indices(i, 2), 1), vertices(indices(i, 2), 2)));
    }

    auto bvh_obj = Accel(tris, quality);

    return bvh_obj;
}

nb::tuple intersect_bvh(const Accel &bvh_accel, const nb::ndarray<Scalar, nb::shape<-1, 3>> &origins,
                        const nb::ndarray<Scalar, nb::shape<-1, 3>> &directions, const nb::ndarray<Scalar, nb::ndim<1>> &tnear,
                        const nb::ndarray<Scalar, nb::ndim<1>> &tfar, bool calculate_reflections, bool robust = true) {

    auto rays = pack_rays(origins, directions, tnear, tfar);
    size_t num_rays = rays.size();

    auto hit_coords = std::make_unique<std::vector<Scalar>>();
    hit_coords->reserve(num_rays * 3);

    auto hit_reflections = std::make_unique<std::vector<Scalar>>();
    if (calculate_reflections) {
        hit_reflections->reserve(num_rays * 3);
    }

    std::vector<int64_t> tri_ids;
    tri_ids.reserve(num_rays);

    std::vector<Scalar> t_values;
    t_values.reserve(num_rays);

    auto intersect_fn = robust ? intersect_accel<false, true> : intersect_accel<false, false>;

    for (auto ray: rays) {
        auto prim_id = intersect_fn(ray, bvh_accel);
        if (prim_id != INVALID_ID) {
            auto hit = ray.org + ray.dir * ray.tmax;
            hit_coords->push_back(hit[0]);
            hit_coords->push_back(hit[1]);
            hit_coords->push_back(hit[2]);
            tri_ids.push_back(bvh_accel.permutation_map[prim_id]);
            t_values.push_back(ray.tmax);
            if (calculate_reflections) {
                auto hit_tri = bvh_accel.precomputed_tris[prim_id].convert_to_tri();
                auto reflection = hit_reflection(hit_tri, ray.dir);
                hit_reflections->push_back(reflection[0]);
                hit_reflections->push_back(reflection[1]);
                hit_reflections->push_back(reflection[2]);
            }
        } else {
            hit_coords->push_back(ScalarNAN);
            hit_coords->push_back(ScalarNAN);
            hit_coords->push_back(ScalarNAN);
            tri_ids.push_back(-1);
            t_values.push_back(ScalarNAN);
            if (calculate_reflections) {
                hit_reflections->push_back(ScalarNAN);
                hit_reflections->push_back(ScalarNAN);
                hit_reflections->push_back(ScalarNAN);
            }
        }
    }

    auto nd_hit_coord = nb::ndarray<nb::numpy, Scalar, nb::shape<-1, 3>>(hit_coords->data(),
                                                                         {num_rays, 3});
    if (calculate_reflections) {

        auto nd_hit_reflections = nb::ndarray<nb::numpy, Scalar, nb::shape<-1, 3>>(hit_reflections->data(),
                                                                                   {num_rays, 3});
        return nb::make_tuple(nd_hit_coord, tri_ids, t_values, nd_hit_reflections);
    } else {
        return nb::make_tuple(nd_hit_coord, tri_ids, t_values);
    }
}

std::vector<bool> occlude_bvh(const Accel &bvh_accel, const nb::ndarray<Scalar, nb::shape<-1, 3>> &origins,
                              const nb::ndarray<Scalar, nb::shape<-1, 3>> &directions, const nb::ndarray<Scalar, nb::ndim<1>> &tnear,
                              const nb::ndarray<Scalar, nb::ndim<1>> &tfar, bool robust = true)
{


    auto rays = pack_rays(origins, directions, tnear, tfar);
    size_t num_rays = rays.size();

    std::vector<bool> results;
    results.reserve(num_rays);
    auto intersect_fn = robust ? intersect_accel<true, true> : intersect_accel<true, false>;

    for (auto ray : rays)
    {
        auto prim_id = intersect_fn(ray, bvh_accel);
        results.push_back(prim_id != INVALID_ID);
    }
    return results;
}


NB_MODULE(_bvh_bind_ext, m) {
    m.doc() = "This is a \"hello world\" example with nanobind3";
    nb::class_<Accel>(m, "Accel")
        .def(nb::init<const std::vector<Tri> &, const std::string &>());
    m.def("build_bvh", &build_bvh, "vertices"_a, "indices"_a, "quality"_a = "medium");
    m.def("intersect_bvh", &intersect_bvh, "bvh_accel"_a, "origins"_a, "directions"_a, "tnear"_a,
          "tfar"_a, "calculate_reflections"_a,"robust"_a = true);
    m.def("occlude_bvh", &occlude_bvh, "bvh_accel"_a, "origins"_a, "directions"_a, "tnear"_a,"tfar"_a,
          "robust"_a = false);
}