//
// Created by Dag Wästberg on 2024-10-02.
//

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
                        const nb::ndarray<Scalar, nb::shape<-1, 3>> &directions, Scalar tmin = 0,
                        Scalar tmax = std::numeric_limits<Scalar>::max(), bool robust = true)
{
    static constexpr size_t invalid_id = std::numeric_limits<size_t>::max();

    auto rays = pack_rays(origins, directions, tmin, tmax);
    size_t num_rays = rays.size();

    auto *hit_coords = new std::vector<Scalar>();
    hit_coords->reserve(num_rays * 3);

    std::vector<int64_t> tri_ids;
    tri_ids.reserve(num_rays);

    std::vector<Scalar> t_values;

    auto intersect_fn = intersect_accel<false, false>;
    if (robust)
        intersect_fn = intersect_accel<false, true>;

    for (auto ray : rays)
    {
        auto prim_id = intersect_fn(ray, bvh_accel);
        if (prim_id != invalid_id)
        {
            auto hit = ray.org + ray.dir * ray.tmax;
            hit_coords->push_back(hit[0]);
            hit_coords->push_back(hit[1]);
            hit_coords->push_back(hit[2]);
            tri_ids.push_back(bvh_accel.permutation_map[prim_id]);
            t_values.push_back(ray.tmax);
        }
        else
        {
            hit_coords->push_back(ScalarNAN);
            hit_coords->push_back(ScalarNAN);
            hit_coords->push_back(ScalarNAN);
            tri_ids.push_back(-1);
            t_values.push_back(ScalarNAN);
        }
    }
    // Delete 'data' when the 'owner' capsule expires
    nb::capsule owner(hit_coords, [](void *p) noexcept
                      { delete static_cast<std::vector<Scalar> *>(p); });
    auto nd_hit_coord = nb::ndarray<nb::numpy, Scalar, nb::shape<-1, 3>>(hit_coords->data(),
                                                                         {num_rays, 3}, owner);
    return nb::make_tuple(nd_hit_coord, tri_ids, t_values);
}

std::vector<bool> occlude_bvh(const Accel &bvh_accel, const nb::ndarray<Scalar, nb::shape<-1, 3>> &origins,
                              const nb::ndarray<Scalar, nb::shape<-1, 3>> &directions, Scalar tmin = 0,
                              Scalar tmax = std::numeric_limits<Scalar>::max(), bool robust = true)
{
    static constexpr size_t invalid_id = std::numeric_limits<size_t>::max();

    auto rays = pack_rays(origins, directions, tmin, tmax);
    size_t num_rays = rays.size();

    std::vector<bool> results;
    results.reserve(num_rays);
    auto intersect_fn = intersect_accel<true, true>;
    if (!robust)
        intersect_fn = intersect_accel<true, false>;
    for (auto ray : rays)
    {
        auto prim_id = intersect_fn(ray, bvh_accel);
        results.push_back(prim_id != invalid_id);
    }
    return results;
}

NB_MODULE(_bvh_bind_ext, m)
{
    m.doc() = "This is a \"hello world\" example with nanobind3";
    nb::class_<Accel>(m, "Accel")
        .def(nb::init<const std::vector<Tri> &, const std::string &>());
    m.def("build_bvh", &build_bvh, "vertices"_a, "indices"_a, "quality"_a = "medium");
    m.def("intersect_bvh", &intersect_bvh, "bvh_accel"_a, "origins"_a, "directions"_a, "tmin"_a = 0,
          "tmax"_a = std::numeric_limits<Scalar>::max(), "robust"_a = false);
    m.def("occlude_bvh", &occlude_bvh, "bvh_accel"_a, "origins"_a, "directions"_a, "tmin"_a = 0,
          "tmax"_a = std::numeric_limits<Scalar>::max(), "robust"_a = false);
}