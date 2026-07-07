// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SketchSort contributors

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstring>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>
#include <stdexcept>
#include <utility>

#include "sketch_sort.hpp"
#include "sketch_sort_minmax.hpp"

namespace py = pybind11;

static py::array search_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> X,
    float cos_dist,
    unsigned int ham_dist,
    unsigned int num_blocks,
    unsigned int num_chunks,
    bool auto_mode,
    float missing_ratio,
    bool centering,
    unsigned int seed,
    bool verbose)
{
    if (X.ndim() != 2) {
        throw std::invalid_argument("X must be a 2-D array of shape (N, D)");
    }
    const auto N = static_cast<std::size_t>(X.shape(0));
    const auto D = static_cast<std::size_t>(X.shape(1));

    std::vector<sketchsort::Pair> pairs;
    {
        sketchsort::SketchSort ss;
        py::gil_scoped_release release;
        ss.search(X.data(), N, D,
                  num_blocks, ham_dist, cos_dist, num_chunks,
                  auto_mode, missing_ratio, centering, seed, verbose,
                  pairs);
    }

    py::array_t<sketchsort::Pair> out(static_cast<py::ssize_t>(pairs.size()));
    if (!pairs.empty()) {
        std::memcpy(out.mutable_data(), pairs.data(),
                    pairs.size() * sizeof(sketchsort::Pair));
    }
    return std::move(out);
}

static void run_from_file_py(
    const std::string &input_path,
    const std::string &output_path,
    float cos_dist,
    unsigned int ham_dist,
    unsigned int num_blocks,
    unsigned int num_chunks,
    bool auto_mode,
    float missing_ratio,
    bool centering,
    unsigned int seed,
    bool verbose)
{
    sketchsort::SketchSort ss;
    py::gil_scoped_release release;
    ss.run(input_path.c_str(), output_path.c_str(),
           num_blocks, ham_dist, cos_dist, num_chunks,
           auto_mode, missing_ratio, centering, seed, verbose);
}

static py::array search_minmax_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> X,
    float minmax_dist,
    unsigned int ham_dist,
    unsigned int num_blocks,
    unsigned int num_chunks,
    bool auto_mode,
    float missing_ratio,
    bool z_normalization,
    bool minmax_normalization,
    long seed,
    bool verbose)
{
    if (X.ndim() != 2) {
        throw std::invalid_argument("X must be a 2-D array of shape (N, D)");
    }
    const auto N = static_cast<std::size_t>(X.shape(0));
    const auto D = static_cast<std::size_t>(X.shape(1));

    std::vector<sketchsort::minmax::Pair> pairs;
    {
        sketchsort::minmax::SketchSort ss;
        py::gil_scoped_release release;
        ss.search(X.data(), N, D,
                  num_blocks, ham_dist, minmax_dist, num_chunks,
                  auto_mode, missing_ratio, z_normalization, minmax_normalization,
                  seed, verbose, pairs);
    }

    py::array_t<sketchsort::minmax::Pair> out(static_cast<py::ssize_t>(pairs.size()));
    if (!pairs.empty()) {
        std::memcpy(out.mutable_data(), pairs.data(),
                    pairs.size() * sizeof(sketchsort::minmax::Pair));
    }
    return std::move(out);
}

static void run_from_file_minmax_py(
    const std::string &input_path,
    const std::string &output_path,
    float minmax_dist,
    unsigned int ham_dist,
    unsigned int num_blocks,
    unsigned int num_chunks,
    bool auto_mode,
    float missing_ratio,
    bool z_normalization,
    bool minmax_normalization,
    long seed,
    bool verbose)
{
    sketchsort::minmax::SketchSort ss;
    py::gil_scoped_release release;
    ss.run(input_path.c_str(), output_path.c_str(),
           num_blocks, ham_dist, minmax_dist, num_chunks,
           auto_mode, missing_ratio, z_normalization, minmax_normalization,
           seed, verbose);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "SketchSort: fast all-pairs similarity search via random projection sketches (cosine and min-max metrics).";

    PYBIND11_NUMPY_DTYPE(sketchsort::Pair, id1, id2, cos_dist);
    PYBIND11_NUMPY_DTYPE(sketchsort::minmax::Pair, id1, id2, minmax_dist);

    m.def("search", &search_py,
          py::arg("X"),
          py::arg("cos_dist")      = 0.01f,
          py::arg("ham_dist")      = 1u,
          py::arg("num_blocks")    = 4u,
          py::arg("num_chunks")    = 3u,
          py::arg("auto_mode")     = false,
          py::arg("missing_ratio") = 0.0001f,
          py::arg("centering")     = false,
          py::arg("seed")          = 0u,
          py::arg("verbose")       = false,
          R"doc(Find all vector pairs in X whose cosine distance is at most cos_dist.

Parameters
----------
X : ndarray, shape (N, D), float32 (other dtypes are cast)
    Input matrix; row i has implicit id i.
cos_dist : float
    Maximum cosine distance for a pair to be reported.
ham_dist : int
    Allowed Hamming distance per sketch chunk.
num_blocks : int
    Number of blocks per chunk used in the multiple-sort enumeration.
num_chunks : int
    Number of independent sketch chunks.
auto_mode : bool
    If True, ham_dist / num_blocks / num_chunks are derived from missing_ratio.
missing_ratio : float
    Target probability of missing a true neighbor (used when auto_mode is True).
centering : bool
    If True, subtract per-dimension mean from X before sketching.
seed : int
    Random seed for the projection RNG (deterministic; default 0).
verbose : bool
    If True, print algorithm progress to stdout/stderr (default False).

Returns
-------
ndarray with dtype [('id1', '<u4'), ('id2', '<u4'), ('cos_dist', '<f4')]
)doc");

    m.def("run_from_file", &run_from_file_py,
          py::arg("input_path"),
          py::arg("output_path"),
          py::arg("cos_dist")      = 0.01f,
          py::arg("ham_dist")      = 1u,
          py::arg("num_blocks")    = 4u,
          py::arg("num_chunks")    = 3u,
          py::arg("auto_mode")     = false,
          py::arg("missing_ratio") = 0.0001f,
          py::arg("centering")     = false,
          py::arg("seed")          = 0u,
          py::arg("verbose")       = false,
          "Read whitespace-separated float vectors from input_path, write 'id1 id2 cos_dist' triples to output_path. Exactly matches the legacy CLI output format.");

    m.def("search_minmax", &search_minmax_py,
          py::arg("X"),
          py::arg("minmax_dist")          = 0.1f,
          py::arg("ham_dist")             = 1u,
          py::arg("num_blocks")           = 4u,
          py::arg("num_chunks")           = 3u,
          py::arg("auto_mode")            = false,
          py::arg("missing_ratio")        = 0.0001f,
          py::arg("z_normalization")      = false,
          py::arg("minmax_normalization") = false,
          py::arg("seed")                 = 0,
          py::arg("verbose")              = false,
          R"doc(Find all vector pairs in X whose min-max (generalized Jaccard) distance is at most minmax_dist.

Sketches are produced by generalized consistent weighted sampling (GCWS).

Parameters
----------
X : ndarray, shape (N, D), float32 (other dtypes are cast)
    Input matrix; row i has implicit id i. May contain negative values.
minmax_dist : float
    Maximum min-max distance (1 - min-max similarity) for a pair to be reported.
ham_dist, num_blocks, num_chunks : int
    Multiple-sort enumeration parameters (used when auto_mode is False).
auto_mode : bool
    If True, ham_dist / num_blocks / num_chunks are derived from missing_ratio.
missing_ratio : float
    Target probability of missing a true neighbor (used when auto_mode is True).
z_normalization : bool
    If True, z-normalize each dimension before sketching.
minmax_normalization : bool
    If True, min-max normalize each dimension to [0, 1] before sketching.
seed : int
    Random seed for the projection RNG. Negative means derive from the clock.
verbose : bool
    If True, print algorithm progress to stdout/stderr (default False).

Returns
-------
ndarray with dtype [('id1', '<u4'), ('id2', '<u4'), ('minmax_dist', '<f4')]
)doc");

    m.def("run_from_file_minmax", &run_from_file_minmax_py,
          py::arg("input_path"),
          py::arg("output_path"),
          py::arg("minmax_dist")          = 0.1f,
          py::arg("ham_dist")             = 1u,
          py::arg("num_blocks")           = 4u,
          py::arg("num_chunks")           = 3u,
          py::arg("auto_mode")            = false,
          py::arg("missing_ratio")        = 0.0001f,
          py::arg("z_normalization")      = false,
          py::arg("minmax_normalization") = false,
          py::arg("seed")                 = 0,
          py::arg("verbose")              = false,
          "Read whitespace-separated float vectors from input_path, write 'id1 id2 minmax_dist' triples to output_path. Matches the standalone min-max CLI output format.");
}
