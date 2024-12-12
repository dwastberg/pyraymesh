#include "types.h"
#include "utils.h"
#include "Accel.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unique_ptr.h>

#include <bvh/v2/vec.h>

#include <bvh/v2/stack.h>
#include <bvh/v2/tri.h>

#include "BS_thread_pool.hpp"

#include <iostream>


namespace nb = nanobind;
using namespace nb::literals;

constexpr Scalar ScalarNAN = std::numeric_limits<Scalar>::quiet_NaN();
constexpr size_t INVALID_ID = std::numeric_limits<size_t>::max();


auto build_bvh(const nb::ndarray<Scalar, nb::shape<-1, 3> > &vertices,
               const nb::ndarray<int, nb::shape<-1, 3> > &indices,
               const std::string &quality = "medium") {
    std::vector<Tri> tris;
    tris.reserve(indices.shape(0));
    for (size_t i = 0; i < indices.shape(0); i++) {
        tris.emplace_back(
            Vec3(vertices(indices(i, 0), 0), vertices(indices(i, 0), 1), vertices(indices(i, 0), 2)),
            Vec3(vertices(indices(i, 1), 0), vertices(indices(i, 1), 1), vertices(indices(i, 1), 2)),
            Vec3(vertices(indices(i, 2), 0), vertices(indices(i, 2), 1), vertices(indices(i, 2), 2)));
    }

    auto bvh_obj = Accel(tris, quality);

    return bvh_obj;
}

nb::tuple intersect_bvh(const Accel &bvh_accel, const nb::ndarray<Scalar, nb::shape<-1, 3> > &origins,
                        const nb::ndarray<Scalar, nb::shape<-1, 3> > &directions,
                        const nb::ndarray<Scalar, nb::ndim<1> > &tnear,
                        const nb::ndarray<Scalar, nb::ndim<1> > &tfar, bool calculate_reflections, bool robust = true,
                        size_t threads = 1) {
    auto rays = pack_rays(origins, directions, tnear, tfar);
    size_t num_rays = rays.size();

    auto hit_coords = std::make_unique<std::vector<Scalar> >();
    hit_coords->resize(num_rays * 3);

    auto hit_reflections = std::make_unique<std::vector<Scalar> >();
    if (calculate_reflections) {
        hit_reflections->resize(num_rays * 3);
    }

    // std::vector<int64_t> tri_ids;
    auto tri_ids = std::make_unique<std::vector<int64_t> >();
    tri_ids->resize(num_rays);

    auto t_values = std::make_unique<std::vector<Scalar> >();
    t_values->resize(num_rays);


    auto intersect_fn = robust ? intersect_accel<false, true> : intersect_accel<false, false>;

    BS::thread_pool pool(threads);

    pool.detach_loop<size_t>(0, num_rays,
                             [&](size_t i) {
                                 auto ray = rays[i];
                                 auto prim_id = intersect_fn(ray, bvh_accel);
                                 if (prim_id != INVALID_ID) {
                                     auto hit = ray.org + ray.dir * ray.tmax;
                                     (*hit_coords)[i * 3 + 0] = hit[0];
                                     (*hit_coords)[i * 3 + 1] = hit[1];
                                     (*hit_coords)[i * 3 + 2] = hit[2];
                                     (*tri_ids)[i] = bvh_accel.permutation_map[prim_id];
                                     (*t_values)[i] = ray.tmax;
                                     if (calculate_reflections) {
                                         auto hit_tri = bvh_accel.precomputed_tris[prim_id].convert_to_tri();
                                         auto reflection = hit_reflection(hit_tri, ray.dir);
                                         (*hit_reflections)[i * 3 + 0] = reflection[0];
                                         (*hit_reflections)[i * 3 + 1] = reflection[1];
                                         (*hit_reflections)[i * 3 + 2] = reflection[2];
                                     }
                                 } else {
                                     (*hit_coords)[i * 3 + 0] = ScalarNAN;
                                     (*hit_coords)[i * 3 + 1] = ScalarNAN;
                                     (*hit_coords)[i * 3 + 2] = ScalarNAN;
                                     (*tri_ids)[i] = -1;
                                     (*t_values)[i] = ScalarNAN;
                                     if (calculate_reflections) {
                                         (*hit_reflections)[i * 3 + 0] = ScalarNAN;
                                         (*hit_reflections)[i * 3 + 1] = ScalarNAN;
                                         (*hit_reflections)[i * 3 + 2] = ScalarNAN;
                                     }
                                 }
                             });
    pool.wait();


    auto nd_hit_coord = nb::ndarray<nb::numpy, Scalar, nb::shape<-1, 3> >(hit_coords->data(),
                                                                          {num_rays, 3});
    auto nd_tri_ids = nb::ndarray<nb::numpy, int64_t, nb::shape<-1> >(tri_ids->data(), {num_rays});
    auto nd_t_values = nb::ndarray<nb::numpy, Scalar, nb::shape<-1> >(t_values->data(), {num_rays});
    if (calculate_reflections) {
        auto nd_hit_reflections = nb::ndarray<nb::numpy, Scalar, nb::shape<-1, 3> >(hit_reflections->data(),
            {num_rays, 3});
        return nb::make_tuple(nd_hit_coord, nd_tri_ids, nd_t_values, nd_hit_reflections);
    } else {
        return nb::make_tuple(nd_hit_coord, nd_tri_ids, nd_t_values);
    }
}

auto occlude_bvh(const Accel &bvh_accel, const nb::ndarray<Scalar, nb::shape<-1, 3> > &origins,
                 const nb::ndarray<Scalar, nb::shape<-1, 3> > &directions,
                 const nb::ndarray<Scalar, nb::ndim<1> > &tnear,
                 const nb::ndarray<Scalar, nb::ndim<1> > &tfar, bool robust = true, size_t threads = 1) {
    auto rays = pack_rays(origins, directions, tnear, tfar);
    size_t num_rays = rays.size();


    auto results = new uint8_t[num_rays];
    //    auto results = nb::ndarray<uint8_t, nb::numpy, nb::ndim<1>>();
    auto intersect_fn = robust ? intersect_accel<true, true> : intersect_accel<true, false>;

    BS::thread_pool pool(threads);

    pool.detach_loop<size_t>(0, num_rays,
                             [&](size_t i) {
                                 auto ray = rays[i];
                                 auto prim_id = intersect_fn(ray, bvh_accel);
                                 // (*results)[i] = (prim_id != INVALID_ID) ? 1 : 0;
                                 bool hit = (prim_id != INVALID_ID);
                                 uint8_t result = hit ? 1 : 0;
                                 results[i] = result;
                             }
    );
    pool.wait();
    nb::capsule owner(results, [](void *p) noexcept {
        delete[] (uint8_t *) p;
    });
    return nb::ndarray<nb::numpy, uint8_t, nb::ndim<1> >(results, {num_rays}, owner);
    //    auto nd_results = nb::ndarray<nb::numpy, uint8_t, nb::shape<-1>>(results->data(), {num_rays});
    //    return nd_results;
}

auto count_intersections(const Accel &bvh_accel, const nb::ndarray<Scalar, nb::shape<-1, 3> > &origins,
                         const nb::ndarray<Scalar, nb::shape<-1, 3> > &directions,
                         const nb::ndarray<Scalar, nb::ndim<1> > &tnear,
                         const nb::ndarray<Scalar, nb::ndim<1> > &tfar, bool robust = true, size_t threads = 1) {
    const auto rays = pack_rays(origins, directions, tnear, tfar);
    size_t num_rays = rays.size();

    auto results = new int64_t[num_rays];
    BS::thread_pool pool(threads);
    pool.detach_loop<size_t>(0, num_rays,
                             [&](size_t i) {
                                 auto ray = rays[i];
                                 const auto hits = interesection_counter(ray, bvh_accel);
                                 results[i] = hits;
                             }
                             );
    pool.wait();

    nb::capsule owner(results, [](void *p) noexcept {
        delete[] (int64_t *) p;
    });
    return nb::ndarray<nb::numpy, int64_t, nb::ndim<1> >(results, {num_rays}, owner);

}

auto traverse(const Accel &bvh_accel, const nb::ndarray<Scalar, nb::shape<1, 3> > &origins,
                         const nb::ndarray<Scalar, nb::shape<1, 3> > &directions) {
    Ray ray = Ray(Vec3(origins(0, 0), origins(0, 1), origins(0, 2)),
                  Vec3(directions(0, 0), directions(0, 1), directions(0, 2)),
                  0.0, std::numeric_limits<Scalar>::infinity());
    std::vector<size_t> tri_ids;
    static constexpr size_t stack_size = 64;
    bvh::v2::SmallStack<Bvh::Index, stack_size> stack;
    bvh_accel.bvh.intersect<false, true>(ray, bvh_accel.bvh.get_root().index, stack,
                                    [&](size_t begin, size_t end) {
                                        for (size_t i = begin; i < end; ++i) {
                                            tri_ids.push_back(bvh_accel.permutation_map[i]);
                                        }
                                        return false;
                                    });



    return tri_ids;

}

NB_MODULE(_bvh_bind_ext, m) {
    m.doc() = "bindings for bvh and functions for ray intersection";
    nb::class_<Accel>(m, "Accel")
            .def(nb::init<const std::vector<Tri> &, const std::string &>());
    m.def("build_bvh", &build_bvh, "vertices"_a, "indices"_a, "quality"_a = "medium");
    m.def("intersect_bvh", &intersect_bvh, "bvh_accel"_a, "origins"_a, "directions"_a, "tnear"_a,
          "tfar"_a, "calculate_reflections"_a, "robust"_a = true, "threads"_a = 1);
    m.def("occlude_bvh", &occlude_bvh, "bvh_accel"_a, "origins"_a, "directions"_a, "tnear"_a, "tfar"_a,
          "robust"_a = true, "threads"_a = 1);
    m.def("count_intersections", &count_intersections, "bvh_accel"_a, "origins"_a, "directions"_a, "tnear"_a,
          "tfar"_a, "robust"_a = true, "threads"_a = 1);
    m.def("traverse", &traverse, "bvh_accel"_a, "origins"_a, "directions"_a);
}
