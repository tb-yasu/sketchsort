/*
 * sketch_sort_minmax.hpp
 * Copyright (c) 2017 Yasuo Tabei
 *
 * SPDX-License-Identifier: MIT
 * See the LICENSE file in the project root for the full license text.
 *
 * Min-max (generalized Jaccard / Tanimoto on real vectors) all-pairs
 * similarity search. Sketches are produced by generalized consistent
 * weighted sampling (Ping Li, KDD'17); the enumeration itself runs on the
 * shared engine in multi_sort_engine.hpp.
 */

#ifndef SKETCH_SORT_MINMAX_HPP
#define SKETCH_SORT_MINMAX_HPP

// Vendored Boost random (deterministic across platforms, unlike libstdc++/
// libc++ <random> distributions) drives the GCWS projection matrices.
// Include only the specific headers we use: the umbrella "./boost/random.hpp"
// pulls in uniform_on_sphere.hpp, whose std::bind2nd was removed in C++17.
#include "./boost/random/mersenne_twister.hpp"
#include "./boost/random/gamma_distribution.hpp"
#include "./boost/random/uniform_real.hpp"
#include "./boost/random/variate_generator.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <ostream>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <stdint.h>
#include <time.h>

namespace sketchsort {
namespace minmax {

struct Pair {
  std::uint32_t id1;
  std::uint32_t id2;
  float         minmax_dist;
};
static_assert(sizeof(Pair)                == 12, "Pair must be 12 bytes (no padding)");
static_assert(offsetof(Pair, id1)         == 0,  "unexpected layout");
static_assert(offsetof(Pair, id2)         == 4,  "unexpected layout");
static_assert(offsetof(Pair, minmax_dist) == 8,  "unexpected layout");

// Metric-level parameters only; the multiple-sort enumeration state lives in
// engine::MultiSortEngine (multi_sort_engine.hpp).
struct Params {
  std::uint32_t num_blocks;
  std::uint32_t num_chunks;
  std::uint32_t chunk_dist;
  float         minmax_dist;
  unsigned long seed;
  std::ostream *os     = nullptr;
  std::vector<Pair> *pairs = nullptr;
  bool          auto_mode            = false;
  float         missing_ratio        = 0.0001f;
  bool          z_normalization      = false;
  bool          minmax_normalization = false;
  bool          verbose              = true;
};

class SketchSort {
  static constexpr std::uint32_t kProjectDim = 32;

 public:
  SketchSort() : sig_pool_(nullptr), dim_(0) {}

  // File in, file out. Matches the standalone CLI's behaviour and output.
  void run(const char *input_path, const char *output_path,
	   std::uint32_t num_blocks,
	   std::uint32_t ham_dist,
	   float         minmax_dist,
	   std::uint32_t num_chunks,
	   bool          auto_mode,
	   float         missing_ratio,
	   bool          z_normalization,
	   bool          minmax_normalization,
	   long          seed,
	   bool          verbose = true);

  // Dense matrix in, pairs out (in-memory). Throws std::invalid_argument /
  // std::runtime_error on bad input instead of calling exit().
  void search(const float *data, std::size_t n_rows, std::size_t n_cols,
	      std::uint32_t num_blocks,
	      std::uint32_t ham_dist,
	      float         minmax_dist,
	      std::uint32_t num_chunks,
	      bool          auto_mode,
	      float         missing_ratio,
	      bool          z_normalization,
	      bool          minmax_normalization,
	      long          seed,
	      bool          verbose,
	      std::vector<Pair> &out);

 private:
  void read_features(const char *input_path);
  void set_features_from_raw(const float *data, std::size_t n_rows, std::size_t n_cols);
  Params init_params(std::uint32_t num_blocks,
                     std::uint32_t ham_dist,
                     float         minmax_dist,
                     std::uint32_t num_chunks,
                     bool          auto_mode,
                     float         missing_ratio,
                     bool          z_normalization,
                     bool          minmax_normalization,
                     long          seed,
                     bool          verbose);
  void run_core(Params &param);
  void project_vectors(std::uint32_t project_dim, std::vector<uint8_t*> &sig,
                       unsigned long seed, bool verbose);
  float calc_minmax_dist(std::uint32_t id1, std::uint32_t id2);
  static double mismatch_prob(float minmax_dist);

  float safe_log(float val);
  void generate_matrix(std::uint32_t project_dim, unsigned long seed,
                       std::vector<std::vector<float> > &mat_r,
                       std::vector<std::vector<float> > &mat_c,
                       std::vector<std::vector<float> > &mat_b);
  void z_normalize();
  void minmax_normalize();

  uint8_t           *sig_pool_;
  std::vector<std::vector<float> > fvs_;
  std::uint64_t      dim_;
};

}  // namespace minmax
}  // namespace sketchsort

#endif  // SKETCH_SORT_MINMAX_HPP
