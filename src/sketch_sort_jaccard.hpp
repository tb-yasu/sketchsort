/*
 * sketch_sort_jaccard.hpp
 * Copyright (c) 2011 Yasuo Tabei
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * See the LICENSE file in the project root for the full license text.
 *
 * Jaccard/Tanimoto all-pairs similarity search over sparse integer-id sets.
 * Sketches are MinHash values (one byte per hash); the enumeration itself
 * runs on the shared engine in multi_sort_engine.hpp.
 */

#ifndef SKETCH_SORT_JACCARD_HPP
#define SKETCH_SORT_JACCARD_HPP

#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <fstream>
#include <sstream>
#include <ostream>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <time.h>
#include <random>

namespace sketchsort {
namespace jaccard {

struct Pair {
  std::uint32_t id1;
  std::uint32_t id2;
  float         jaccard_dist;
};
static_assert(sizeof(Pair)                 == 12, "Pair must be 12 bytes (no padding)");
static_assert(offsetof(Pair, id1)          == 0,  "unexpected layout");
static_assert(offsetof(Pair, id2)          == 4,  "unexpected layout");
static_assert(offsetof(Pair, jaccard_dist) == 8,  "unexpected layout");

// Metric-level parameters only; the multiple-sort enumeration state lives in
// engine::MultiSortEngine (multi_sort_engine.hpp).
struct Params {
  unsigned int num_blocks;
  unsigned int num_chunks;
  unsigned int chunk_dist;
  float         jaccard_dist;
  std::ostream *os    = nullptr;
  std::vector<Pair> *pairs = nullptr;
  bool          auto_mode     = false;
  float         missing_ratio = 0.0001f;
  bool          verbose       = true;
};

class SketchSort {
  static constexpr unsigned int kProjectDim = 32;

  std::vector<uint8_t> sig_storage_;
  std::mt19937  rng_;
  std::vector<std::vector<std::uint32_t> > fvs_;
  int                max_item_;

  void read_features(const char *file_name);
  void set_features_from_sets(const std::vector<std::vector<std::uint32_t> > &sets);
  Params init_params(unsigned int num_blocks,
                     unsigned int ham_dist,
                     float        jaccard_dist,
                     unsigned int num_chunks,
                     bool         auto_mode,
                     float        missing_ratio,
                     unsigned int seed,
                     bool         verbose);
  void run_core(Params &param);
  void project_vectors(unsigned int project_dim, std::vector<uint8_t*> &sig);
  float calc_jaccard_dist(unsigned int id1, unsigned int id2);
  float check_upper_bound(unsigned int id1, unsigned int id2);
  static double mismatch_prob(float jaccard_dist);
 public:
  SketchSort() : max_item_(0) {}

  // File in, file out. Matches the standalone CLI's behaviour and output.
  void run(const char *file_name, const char *output_name,
	   unsigned int num_blocks,
	   unsigned int ham_dist,
	   float        jaccard_dist,
	   unsigned int num_chunks,
	   bool         auto_mode,
	   float        missing_ratio,
	   unsigned int seed,
	   bool         verbose = true);

  // Sparse integer-id sets in, pairs out (in-memory). Row i has id i (empty
  // sets are kept so ids match the caller's input indices). Throws
  // std::invalid_argument on bad input instead of calling exit().
  void search(const std::vector<std::vector<std::uint32_t> > &sets,
	      unsigned int num_blocks,
	      unsigned int ham_dist,
	      float        jaccard_dist,
	      unsigned int num_chunks,
	      bool         auto_mode,
	      float        missing_ratio,
	      unsigned int seed,
	      bool         verbose,
	      std::vector<Pair> &out);
};

}  // namespace jaccard
}  // namespace sketchsort

#endif // SKETCH_SORT_JACCARD_HPP
