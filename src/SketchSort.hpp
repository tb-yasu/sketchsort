// SPDX-License-Identifier: MIT
/*
 * SketchSort.hpp
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

#ifndef SKETCHSORT_HPP
#define SKETCHSORT_HPP

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

using namespace boost::numeric::ublas;  // boost::numeric::ublas
using namespace boost::numeric;         // boost::numeric

#ifndef M_PI
#define M_PI 3.14159265358979323846/* pi */
#define M_PI_2 1.57079632679489661923/* pi/2 */
#define M_PI_4 0.78539816339744830962/* pi/4 */
#endif

struct pstat {
  int start;
  int end;
};

namespace sketchsort_api {
struct Pair {
  std::uint32_t id1;
  std::uint32_t id2;
  float         cos_dist;
};
static_assert(sizeof(Pair)             == 12, "Pair must be 12 bytes (no padding)");
static_assert(offsetof(Pair, id1)      == 0,  "unexpected layout");
static_assert(offsetof(Pair, id2)      == 4,  "unexpected layout");
static_assert(offsetof(Pair, cos_dist) == 8,  "unexpected layout");
}

struct params {
  unsigned int numblocks;
  unsigned int numchunks;
  unsigned int projectDim;
  unsigned int chunk_dist;
  unsigned int chunks;
  unsigned int num_seq;
  unsigned int seq_len;
  unsigned int chunk_len;
  unsigned int start_chunk;
  unsigned int end_chunk;
  unsigned int cchunk;
  unsigned int *counter;
  pstat        *pos;
  pstat        *pchunks;
  float         cosDist;
  std::vector<unsigned int> ids;
  std::vector<int> blocks;
  std::ostream *os    = nullptr;
  std::vector<sketchsort_api::Pair> *pairs = nullptr;
  bool         autoFlag     = false;
  float        missingratio = 0.0001f;
  bool         centering    = false;
  unsigned int seed         = 0;
  bool         verbose      = true;
};

class SketchSort {
  boost::pool<> *p;
  std::vector<boost::numeric::ublas::vector<float> > fvs;
  std::vector<float> norms;
  uint8_t num_char;
  unsigned int dim;

  uint64_t numSort;
  uint64_t numHamDist;
  uint64_t numCosDist;

  void readFeature(const char *fname);
  void setFeaturesFromRaw(const float *data, std::size_t n_rows, std::size_t n_cols);
  void centeringData();
  void preComputeNorms();
  void runCore(params &param);
  int projectVectors(unsigned int projectDim, std::vector<uint8_t*> &sig, params &param);
  void report(std::vector<uint8_t*> &sig, int l, int r, params &param);
  void sort(std::vector<uint8_t*> &sig, int spos, int epos, int l, int r, params &param);
  void radixsort(std::vector<uint8_t*> &sig, int spos, int epos, int l, int r, params &param);
  void insertionSort(std::vector<uint8_t*> &sig, int spos, int epos, int l, int r, params &param);
  bool calc_chunk_hamdist(uint8_t *seq1, uint8_t *seq2, const params &param);
  bool calc_hamdist(uint8_t *seq1, uint8_t *seq2, const params &param);
  bool check_canonical(uint8_t *seq1, uint8_t *seq2, const params &param);
  bool check_chunk_canonical(uint8_t *seq1, uint8_t *seq2, const params &param);
  float checkCos(unsigned int id1, unsigned int id2);
  double calcMissingEdgeRatio(params &param);
  void classify(std::vector<uint8_t*> &sig, int spos, int epos, int l, int r, int bpos, params &param);
  void multi_classification(std::vector<uint8_t*> &sig, int maxind, int l, int r, params &param);
  void refinement();
  void insertKnnList(unsigned int from_id, unsigned int to_id, float cosDist);
  void decideParameters(float _missingratio, params &param);
 public:
  SketchSort() {};
  void run(const char *fname, const char *oname,
	   unsigned int _numblocks,
           unsigned int _dist,
	   float        _cosDist,
	   unsigned int _numchunks,
	   bool         _autoFlag,
	   float        _missingratio,
	   bool         _centering,
	   unsigned int _seed    = 0,
	   bool         _verbose = true);

  void search(const float *data, std::size_t n_rows, std::size_t n_cols,
	      unsigned int _numblocks,
	      unsigned int _dist,
	      float        _cosDist,
	      unsigned int _numchunks,
	      bool         _autoFlag,
	      float        _missingratio,
	      bool         _centering,
	      unsigned int _seed,
	      bool         _verbose,
	      std::vector<sketchsort_api::Pair> &out);
};

#endif // SKETCHSORT_HPP
