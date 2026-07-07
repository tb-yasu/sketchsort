/*
 * sketch_sort_jaccard.hpp
 * Copyright (c) 2011 Yasuo Tabei
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * See the LICENSE file in the project root for the full license text.
 *
 * Jaccard/Tanimoto all-pairs similarity search over sparse integer-id sets.
 * Sketches are MinHash values (one byte per hash). Enumeration reuses the
 * same multiple-sort machinery as the cosine and min-max variants but lives
 * in its own sub-namespace so the three cores can coexist in one library.
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

struct PosRange {
  int start;
  int end;
};

struct Pair {
  std::uint32_t id1;
  std::uint32_t id2;
  float         jaccard_dist;
};
static_assert(sizeof(Pair)                 == 12, "Pair must be 12 bytes (no padding)");
static_assert(offsetof(Pair, id1)          == 0,  "unexpected layout");
static_assert(offsetof(Pair, id2)          == 4,  "unexpected layout");
static_assert(offsetof(Pair, jaccard_dist) == 8,  "unexpected layout");

struct Params {
  unsigned int num_blocks;
  unsigned int num_chunks;
  unsigned int project_dim;
  unsigned int chunk_dist;
  unsigned int num_seq;
  unsigned int seq_len;
  unsigned int chunk_len;
  unsigned int start_chunk;
  unsigned int cchunk;
  unsigned int *counter;
  PosRange     *pos;
  PosRange     *pchunks;
  float         jaccard_dist;
  std::vector<unsigned int> ids;
  std::vector<int> blocks;
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
  std::uint32_t      num_char_;

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
  int project_vectors(unsigned int project_dim, std::vector<uint8_t*> &sig, Params &param);
  void report(std::vector<uint8_t*> &sig, int l, int r, Params &param);
  void radix_sort(std::vector<uint8_t*> &sig, int spos, int epos, int l, int r, Params &param);
  void insertion_sort(std::vector<uint8_t*> &sig, int spos, int epos, int l, int r, Params &param);
  bool calc_chunk_hamdist(uint8_t *seq1, uint8_t *seq2, const Params &param);
  bool check_canonical(uint8_t *seq1, uint8_t *seq2, const Params &param);
  bool check_chunk_canonical(uint8_t *seq1, uint8_t *seq2, const Params &param);
  float calc_jaccard_dist(unsigned int id1, unsigned int id2);
  float check_upper_bound(unsigned int id1, unsigned int id2);
  double calc_missing_edge_ratio(Params &param);
  void classify(std::vector<uint8_t*> &sig, int spos, int epos, int l, int r, int bpos, Params &param);
  void decide_parameters(float missing_ratio, Params &param);
  void multi_classification(std::vector<uint8_t*> &sig, int maxind, int l, int r, Params &param);
 public:
  SketchSort() : max_item_(0), num_char_(0) {}

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
