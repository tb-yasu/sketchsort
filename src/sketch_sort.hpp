// SPDX-License-Identifier: MIT
/*
 * sketch_sort.hpp
 * Copyright (c) 2011 Yasuo Tabei All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE and * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef SKETCH_SORT_HPP
#define SKETCH_SORT_HPP

// for boost
#include "./boost/pool/pool.hpp"
#include "./boost/pool/object_pool.hpp"
#include "./boost/numeric/ublas/matrix.hpp"
#include "./boost/numeric/ublas/triangular.hpp"
#include "./boost/numeric/ublas/symmetric.hpp"
#include "./boost/numeric/ublas/hermitian.hpp"
#include "./boost/numeric/ublas/matrix_sparse.hpp"
#include "./boost/numeric/ublas/vector.hpp"
#include "./boost/numeric/ublas/vector_sparse.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <cmath>
#include <cstring>
#include <iterator>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <set>
#include <stdint.h>
#include <time.h>
#include <limits>
#include <ostream>
#include <stdexcept>

#include "./boost/random/mersenne_twister.hpp"
#include "./boost/random/normal_distribution.hpp"
#include "./boost/random/variate_generator.hpp"
#include "./boost/random/uniform_real.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846/* pi */
#define M_PI_2 1.57079632679489661923/* pi/2 */
#define M_PI_4 0.78539816339744830962/* pi/4 */
#endif

namespace sketchsort {

using namespace boost::numeric::ublas;  // boost::numeric::ublas
using namespace boost::numeric;         // boost::numeric

struct PosRange {
  int start;
  int end;
};

struct Pair {
  std::uint32_t id1;
  std::uint32_t id2;
  float         cos_dist;
};
static_assert(sizeof(Pair)             == 12, "Pair must be 12 bytes (no padding)");
static_assert(offsetof(Pair, id1)      == 0,  "unexpected layout");
static_assert(offsetof(Pair, id2)      == 4,  "unexpected layout");
static_assert(offsetof(Pair, cos_dist) == 8,  "unexpected layout");

struct Params {
  unsigned int num_blocks;
  unsigned int num_chunks;
  unsigned int project_dim;
  unsigned int ham_dist;
  unsigned int num_sketches;
  unsigned int sketch_len;
  unsigned int chunk_len;
  unsigned int chunk_start;
  unsigned int chunk_end;
  unsigned int current_chunk;
  unsigned int *radix_counts;
  PosRange     *block_ranges;
  PosRange     *chunk_ranges;
  float         cos_dist;
  std::vector<unsigned int> row_ids;
  std::vector<int> selected_blocks;
  std::ostream *out_stream = nullptr;
  std::vector<Pair> *pairs = nullptr;
  bool         auto_mode     = false;
  float        missing_ratio = 0.0001f;
  bool         centering     = false;
  unsigned int seed          = 0;
  bool         verbose       = true;
};

class SketchSort {
  static constexpr unsigned int kProjectDim = 32;

  boost::pool<> *sketch_pool_;
  std::vector<boost::numeric::ublas::vector<float> > feature_vectors_;
  std::vector<float> inv_norms_;
  uint8_t alphabet_size_;
  unsigned int dim_;

  uint64_t num_ham_dist_;
  uint64_t num_cos_dist_;

  void read_features(const char *input_path);
  void set_features_from_raw(const float *data, std::size_t n_rows, std::size_t n_cols);
  void center_data();
  void precompute_norms(bool centered);
  void run_core(Params &param);
  void project_vectors(unsigned int project_dim, std::vector<uint8_t*> &sketches, Params &param);
  void report(std::vector<uint8_t*> &sketches, int left, int right, Params &param);
  void sort(std::vector<uint8_t*> &sketches, int start_pos, int end_pos, int left, int right, Params &param);
  void radix_sort(std::vector<uint8_t*> &sketches, int start_pos, int end_pos, int left, int right, Params &param);
  void insertion_sort(std::vector<uint8_t*> &sketches, int start_pos, int end_pos, int left, int right, Params &param);
  bool check_chunk_ham_dist(uint8_t *sketch1, uint8_t *sketch2, const Params &param);
  bool check_canonical(uint8_t *sketch1, uint8_t *sketch2, const Params &param);
  bool check_chunk_canonical(uint8_t *sketch1, uint8_t *sketch2, const Params &param);
  float calc_cos_dist(unsigned int id1, unsigned int id2);
  double calc_missing_edge_ratio(Params &param);
  void classify(std::vector<uint8_t*> &sketches, int start_pos, int end_pos, int left, int right, int block_idx, Params &param);
  void multi_classification(std::vector<uint8_t*> &sketches, int start_block, int left, int right, Params &param);
  void decide_parameters(float missing_ratio, Params &param);
  Params init_params(unsigned int num_blocks,
                      unsigned int ham_dist,
                      float        cos_dist,
                      unsigned int num_chunks,
                      bool         auto_mode,
                      float        missing_ratio,
                      bool         centering,
                      unsigned int seed,
                      bool         verbose);
 public:
  SketchSort() : sketch_pool_(nullptr), alphabet_size_(0), dim_(0), num_ham_dist_(0), num_cos_dist_(0) {}
  void run(const char *input_path, const char *output_path,
	   unsigned int num_blocks,
           unsigned int ham_dist,
	   float        cos_dist,
	   unsigned int num_chunks,
	   bool         auto_mode,
	   float        missing_ratio,
	   bool         centering,
	   unsigned int seed    = 0,
	   bool         verbose = true);

  void search(const float *data, std::size_t n_rows, std::size_t n_cols,
	      unsigned int num_blocks,
	      unsigned int ham_dist,
	      float        cos_dist,
	      unsigned int num_chunks,
	      bool         auto_mode,
	      float        missing_ratio,
	      bool         centering,
	      unsigned int seed,
	      bool         verbose,
	      std::vector<Pair> &out);
};

} // namespace sketchsort

#endif // SKETCH_SORT_HPP
