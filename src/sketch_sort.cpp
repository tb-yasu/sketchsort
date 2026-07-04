// SPDX-License-Identifier: MIT
/*
 * sketch_sort.cpp
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

#include "sketch_sort.hpp"

namespace sketchsort {

template<class T>
static inline uint8_t sign(T val) {
  if (val > 0)
    return 1;
  return 0;
}

// Parse one whitespace-delimited token as a float. A failed `stream >> float`
// may or may not set eofbit depending on the C++ standard library (libc++
// scans hex-float chars, so line-final garbage like "abc" slipped past the
// old `!is.eof()` check on some platforms); token-wise strtof parsing is
// deterministic. Rejects trailing garbage, inf/nan, and overflow, matching
// what num_get accepted.
static float parse_float_token(const std::string &token, unsigned int line_no) {
  char *end = nullptr;
  float v = std::strtof(token.c_str(), &end);
  if (end != token.c_str() + token.size() || !std::isfinite(v)) {
    std::ostringstream msg;
    msg << "line " << line_no << " contains a non-numeric token";
    throw std::runtime_error(msg.str());
  }
  return v;
}

void SketchSort::read_features(const char *input_path) {
  std::ifstream ifs(input_path);

  if (!ifs) {
    throw std::runtime_error(std::string("cannot open input file: ") + input_path);
  }

  feature_vectors_.clear();
  dim_            = 0;
  std::string line;
  unsigned int line_no = 0;
  while (std::getline(ifs, line)) {
    ++line_no;
    if (line.find_first_not_of(" \t\r\n\v\f") == std::string::npos) {
      std::ostringstream msg;
      msg << "line " << line_no << " is empty or contains only whitespace";
      throw std::runtime_error(msg.str());
    }

    feature_vectors_.resize(feature_vectors_.size() + 1);
    boost::numeric::ublas::vector<float> &vec = feature_vectors_[feature_vectors_.size() - 1];
    uint32_t num_vals = 0;
    std::istringstream is(line);
    std::string token;
    if (dim_ != 0) {
      vec.resize(dim_);
      while (is >> token) {
	if (num_vals >= dim_) {
	  std::ostringstream msg;
	  msg << "line " << line_no << " has more than " << dim_
	      << " values, expected dimension " << dim_;
	  throw std::runtime_error(msg.str());
	}
	vec[num_vals++] = parse_float_token(token, line_no);
      }
      if (num_vals !=  dim_) {
	std::ostringstream msg;
	msg << "dimensions of the input vector should be the same. "
	    << "expected " << dim_ << " got " << num_vals
	    << " on line: " << line;
	throw std::runtime_error(msg.str());
      }
    }
    else {
      while (is >> token) {
	vec.resize(num_vals + 1);
	vec[num_vals] = parse_float_token(token, line_no);
	num_vals++;
      }
      dim_ = num_vals;
    }
  }

  if (feature_vectors_.empty()) {
    throw std::runtime_error("input file contains no data rows");
  }
}

void SketchSort::set_features_from_raw(const float *data, std::size_t n_rows, std::size_t n_cols) {
  feature_vectors_.clear();
  feature_vectors_.resize(n_rows);
  for (std::size_t i = 0; i < n_rows; i++) {
    boost::numeric::ublas::vector<float> &vec = feature_vectors_[i];
    vec.resize(n_cols);
    const float *row = data + i * n_cols;
    for (std::size_t j = 0; j < n_cols; j++) {
      vec[j] = row[j];
    }
  }
  dim_ = static_cast<unsigned int>(n_cols);
}

void SketchSort::center_data() {
  size_t num_dims = feature_vectors_[0].size();
  size_t num_rows = feature_vectors_.size();
  float  mean;
  for (size_t i = 0; i < num_dims; i++) {
    mean = 0.f;
    for (size_t j = 0; j < num_rows; j++) {
      mean += feature_vectors_[j][i];
    }
    mean /= (float)num_rows;
    for (size_t j = 0; j < num_rows; j++) {
      feature_vectors_[j][i] -= mean;
    }
  }
}

/* sparce random projection
int SketchSort::project_vectors(unsigned int project_dim, std::vector<uint8_t*> &sketches, Params &param) {

  sketch_pool_ = new boost::pool<>(sizeof(uint8_t));
  sketches.resize(feature_vectors_.size());
  param.row_ids.resize(feature_vectors_.size());
  for (size_t i = 0; i < sketches.size(); i++) {
    //    sketches[i]    = new uint32_t[project_dim + 1];
    sketches[i]    = (uint8_t*)sketch_pool_->ordered_malloc(project_dim + 1);
    param.row_ids[i] = i;
  }

  boost::mt19937 gen(static_cast<unsigned long>(time(0)));
  boost::uniform_real<> normal_dist(0.f, 1.f);
  boost::variate_generator<boost::mt19937&, boost::uniform_real<> > rand(gen, normal_dist);
  //  double tiny = 1.0/1.79e+308;
  std::vector<std::pair<int, float> > rand_vec;
  float s = sqrt(float(dim_));
  //  float s     = dim_/log(dim_);
  float thr   = 1.f/(2*s);
  float coff  = sqrt(s);
  for (size_t i = 0; i < project_dim; i++) {
    rand_vec.clear();
    for (size_t j = 0; j < dim_; j++) {
      float r   = rand();
      if       (r < thr) {
        rand_vec.push_back(std::make_pair(j, coff));
      } else if (r < 2*thr) {
        rand_vec.push_back(std::make_pair(j, -coff));
      }
    }

    for (size_t j = 0; j < feature_vectors_.size(); j++) {
      boost::numeric::ublas::vector<float> &vec  = feature_vectors_[j];
      double dot = 0.f;
      for (size_t k = 0; k < rand_vec.size(); k++) {
        dot += vec[rand_vec[k].first] * rand_vec[k].second;
      }
      sketches[j][i+1] = sign(dot);
    }
  }
  param.sketch_len = project_dim;
  param.num_sketches = feature_vectors_.size();

  return 1;
}
*/

void SketchSort::project_vectors(unsigned int project_dim, std::vector<uint8_t*> &sketches, Params &param) {
  std::vector<float> rand_vec;
  sketch_pool_ = new boost::pool<>(sizeof(uint8_t));
  sketches.resize(feature_vectors_.size());
  param.row_ids.resize(feature_vectors_.size());
  for (size_t i = 0; i < sketches.size(); i++) {
    //    sketches[i]    = new uint32_t[project_dim + 1];
    sketches[i]    = (uint8_t*)sketch_pool_->ordered_malloc(project_dim + 1);
    param.row_ids[i] = i;
  }

  boost::mt19937 gen(static_cast<unsigned long>(param.seed));
  boost::normal_distribution<> normal_dist(0.f, 1.f);
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > rand(gen, normal_dist);

  //  double tiny = 1.0/1.79e+308;
  rand_vec.resize(dim_ + 1);
  for (size_t i = 0; i < project_dim; i++) {
    for (size_t j = 0; j <= dim_; j++) {
      rand_vec[j] = rand();
    }

    for (size_t j = 0; j < feature_vectors_.size(); j++) {
      boost::numeric::ublas::vector<float> &vec  = feature_vectors_[j];
      double dot = 0.f;
      for (size_t k = 0; k < vec.size(); k++)
        dot += vec[k] * rand_vec[k];

      sketches[j][i+1] = sign(dot);
    }
  }
  param.sketch_len = project_dim;
  param.num_sketches = feature_vectors_.size();

}

inline float SketchSort::calc_cos_dist(unsigned int id1, unsigned int id2) {
  ++num_cos_dist_;
  boost::numeric::ublas::vector<float> &vec1 = feature_vectors_[id1];
  boost::numeric::ublas::vector<float> &vec2 = feature_vectors_[id2];
  float dot = boost::numeric::ublas::inner_prod(vec1, vec2);

  return (1.f - dot*(inv_norms_[id1]*inv_norms_[id2]));
}

inline void SketchSort::sort(std::vector<uint8_t*> &sketches, int start_pos, int end_pos, int left, int right, Params &param) {
   if (right - left + 1 > 50) radix_sort(sketches, start_pos, end_pos, left, right, param);
  else                        insertion_sort(sketches, start_pos, end_pos, left, right, param);
}

inline void SketchSort::radix_sort(std::vector<uint8_t*> &sketches, int start_pos, int end_pos, int left, int right, Params &param) {
  unsigned int *counts = param.radix_counts;
  std::vector<unsigned int> &row_ids   = param.row_ids;
  std::vector<uint8_t*> new_sketches(right - left + 1);
  std::vector<unsigned int> new_ids(right - left + 1);
  unsigned int dst;
  int byte_pos = start_pos - 1;
  while (++byte_pos <= end_pos) {
    for (int i = 0; i < alphabet_size_; i++) *(counts + i) = 0;
    for (int i = left; i <= right; i++) counts[sketches[i][byte_pos]]++;
    for (int i = 1; i < alphabet_size_; i++) *(counts + i) += *(counts + i - 1);
    for (int i = right; i >= left; --i) {
      dst = --counts[sketches[i][byte_pos]] + left;
      new_ids[dst - left] = row_ids[i];
      new_sketches[dst - left] = sketches[i];
    }
    if (++byte_pos <= end_pos) {
      for (int i = 0; i < alphabet_size_; i++) *(counts + i) = 0;
      for (int i = left; i <= right; i++) counts[new_sketches[i - left][byte_pos]]++;
      for (int i = 1; i < alphabet_size_; i++) *(counts + i) += *(counts + i - 1);
      for (int i = right; i >= left; --i) {
	dst = --counts[new_sketches[i - left][byte_pos]] + left;
	row_ids[dst] = new_ids[i - left];
	sketches[dst] = new_sketches[i - left];
      }
    }
    else {
      for (int i = left; i <= right; i++) {
	row_ids[i] = new_ids[i - left];
	sketches[i] = new_sketches[i - left];
      }
      return;
    }
  }
}

inline void SketchSort::insertion_sort(std::vector<uint8_t*> &sketches, int start_pos, int end_pos, int left, int right, Params &param) {
  int i, j;
  uint8_t *pivot, pivot_val;
  unsigned int pivot_id;
  std::vector<unsigned int> &row_ids = param.row_ids;
  for (int byte_pos = start_pos; byte_pos <= end_pos; byte_pos++) {
    for (i = left + 1; i <= right; i++) {
      pivot = sketches[i]; pivot_val = sketches[i][byte_pos]; pivot_id = row_ids[i];
      for (j = i; j > left && sketches[j-1][byte_pos] > pivot_val; j--) {
	sketches[j]       = sketches[j-1];
	row_ids[j]        = row_ids[j-1];
      }
      sketches[j] = pivot;
      row_ids[j] = pivot_id;
    }
  }
}

inline void SketchSort::classify(std::vector<uint8_t*> &sketches, int start_pos, int end_pos, int left, int right, int block_idx, Params &param) {
  int run_start = left, run_end = right;
  for (int iter = left + 1; iter <= right; iter++) {
    if (!std::equal(sketches[run_start] + start_pos, sketches[run_start] + end_pos + 1, sketches[iter] + start_pos)) {
      run_end = iter - 1;
      if (run_end - run_start >= 1)
	multi_classification(sketches, block_idx + 1, run_start, run_end, param);
      run_start = iter;
    }
  }
  if (right - run_start >= 1)
    multi_classification(sketches, block_idx + 1, run_start, right, param);
}

inline bool SketchSort::check_chunk_ham_dist(uint8_t *sketch1, uint8_t *sketch2, const Params &param) {
  ++num_ham_dist_;
  unsigned int dist = 0;
  for (size_t i = 1;  i <= param.chunk_len; i++)
    if (*sketch1++ != *sketch2++ && ++dist > param.ham_dist) return false;
  return true;
}

inline bool SketchSort::check_chunk_canonical(uint8_t *sketch1, uint8_t *sketch2, const Params &param) {
  unsigned int dist = 0;
  int scan_end   = param.chunk_ranges[param.current_chunk].start - 1;
  int j          = 1;
  int chunk_end  = param.chunk_ranges[j].end;
  int i          = 0;

  while (++i <= scan_end) {
    if ((dist += abs(sketch1[i] - sketch2[i])) > param.ham_dist) {
      while (++i <= chunk_end) dist += abs(sketch1[i] - sketch2[i]);
			    //	if (sketch1[i] != sketch2[i]) ++dist;
      dist = 0;
      chunk_end = param.chunk_ranges[++j].end;
      i    = param.chunk_ranges[j].start - 1;
      continue;
    }
    if (chunk_end == i)
      return false;
  }
  return true;
}

inline bool SketchSort::check_canonical(uint8_t *sketch1, uint8_t *sketch2, const Params &param) {
  size_t start_block = 1, end_block = 1;
  size_t b;
  for (size_t i = 0, size = param.selected_blocks.size(); i < size; i++) {
    end_block = param.selected_blocks[i];
    for (b = start_block; b < end_block; b++) {
      if (std::equal(sketch1 + param.block_ranges[b].start, sketch1 + param.block_ranges[b].end + 1, sketch2 + param.block_ranges[b].start))
	  return false;
    }
    start_block = param.selected_blocks[i] + 1;
  }
  return true;
}


inline void SketchSort::report(std::vector<uint8_t*> &sketches, int left, int right, Params &param) {
  float cos_dist;
  for (int i = left; i < right; i++) {
    for (int j = i + 1; j <= right; j++) {
      if (check_canonical(sketches[i], sketches[j], param) &&
	  check_chunk_ham_dist(sketches[i] + param.chunk_start, sketches[j] + param.chunk_start, param) &&
	  check_chunk_canonical(sketches[i], sketches[j], param) &&
	  ((cos_dist = calc_cos_dist(param.row_ids[i], param.row_ids[j])) <= param.cos_dist)) {
	if (param.pairs) {
	  Pair pr;
	  pr.id1      = static_cast<std::uint32_t>(param.row_ids[i]);
	  pr.id2      = static_cast<std::uint32_t>(param.row_ids[j]);
	  pr.cos_dist = cos_dist;
	  param.pairs->push_back(pr);
	}
	if (param.out_stream) {
	  (*param.out_stream) << param.row_ids[i] << " " << param.row_ids[j] << " " << cos_dist << "\n";
	}
      }
    }
  }
}

void SketchSort::multi_classification(std::vector<uint8_t*> &sketches, int start_block, int left, int right, Params &param) {
  if (param.selected_blocks.size() == param.num_blocks - param.ham_dist) {
    report(sketches, left, right, param);
    return;
  }

  for (int block_idx = start_block; block_idx <= (int)param.num_blocks; block_idx++) {
    if (param.selected_blocks.size() + (param.num_blocks - block_idx + 1) < param.num_blocks - param.ham_dist) { // pruning
      //      std::cerr << "return " << std::endl;
      return;
    }
    param.selected_blocks.push_back(block_idx);
    sort(sketches, param.block_ranges[block_idx].start, param.block_ranges[block_idx].end, left, right, param);
    classify(sketches, param.block_ranges[block_idx].start, param.block_ranges[block_idx].end, left, right, block_idx, param);
    param.selected_blocks.pop_back();
  }
}

static double combination(int n, int m) {
  double sum = 1.0;
  for (int i = 0; i < m; i++) {
    sum *= static_cast<double>(n - i) / static_cast<double>(m - i);
  }
  return sum;
}

double SketchSort::calc_missing_edge_ratio(Params &param) {
  double sum = 0.f;
  double prob = acos(1.0 - param.cos_dist)/M_PI;
  for (unsigned int k = 0; k <= param.ham_dist; k++) {
    sum += (combination(param.project_dim, k) * pow(prob, k) * pow(1 - prob, param.project_dim - k));
  }
  return pow(1.0 - sum, param.num_chunks);
}

void SketchSort::precompute_norms(bool centered) {
  inv_norms_.resize(feature_vectors_.size());
  float sum;
  for (size_t i = 0; i < feature_vectors_.size(); i++) {
    boost::numeric::ublas::vector<float> &vec = feature_vectors_[i];
    sum = 0.f;
    for (size_t j = 0; j < vec.size(); j++) {
      sum += pow(vec[j], 2);
    }
    if (!(std::isfinite(sum) && sum > 0.f)) {
      std::ostringstream msg;
      msg << "row " << i << " has a zero or non-finite norm"
          << (centered ? " (after centering)" : "")
          << "; cannot compute a cosine distance for it";
      throw std::invalid_argument(msg.str());
    }
    inv_norms_[i] = 1.f/sqrt(sum);
  }
}

void SketchSort::decide_parameters(float missing_ratio, Params &param) {
  unsigned int ham_dist   = 1;
  unsigned int num_blocks = ham_dist + 3;
  unsigned int num_chunks = 0;

  do {
    if (num_chunks > 30) {
      ham_dist   += 1;
      num_blocks  = ham_dist + 3;
      num_chunks  = 0;
    }
    num_chunks += 1;
    param.ham_dist   = ham_dist;
    param.num_blocks = num_blocks;
    param.num_chunks = num_chunks;
  } while (calc_missing_edge_ratio(param) >= missing_ratio);
}

void SketchSort::run_core(Params &param) {
  num_cos_dist_ = 0;
  num_ham_dist_ = 0;

  if (param.auto_mode) {
    if (!(param.missing_ratio > 0.f && param.missing_ratio < 1.f)) {
      std::ostringstream msg;
      msg << "missing_ratio must be in (0, 1), got " << param.missing_ratio;
      throw std::invalid_argument(msg.str());
    }
    if (param.verbose) std::cerr << "deciding parameters such that the missing edge ratio is no more than " << param.missing_ratio << std::endl;
    decide_parameters(param.missing_ratio, param);
    if (param.verbose) {
      std::cout << "decided parameters:" << std::endl;
      std::cout << "hamming distance threshold: " << param.ham_dist << std::endl;
      std::cout << "number of blocks: " << param.num_blocks << std::endl;
      std::cout << "number of chunks: "  << param.num_chunks << std::endl;
      std::cout << std::endl;
    }
  }

  if (param.num_chunks < 1) {
    throw std::invalid_argument("num_chunks must be >= 1");
  }
  if (param.num_chunks > (std::numeric_limits<unsigned int>::max)() / param.project_dim) {
    std::ostringstream msg;
    msg << "num_chunks (" << param.num_chunks << ") is too large: "
        << param.project_dim << " * num_chunks would overflow";
    throw std::invalid_argument(msg.str());
  }
  if (param.num_blocks < 1) {
    std::ostringstream msg;
    msg << "num_blocks must be >= 1, got " << param.num_blocks;
    throw std::invalid_argument(msg.str());
  }
  if (param.num_blocks > param.project_dim) {
    std::ostringstream msg;
    if (param.auto_mode) {
      msg << "cos_dist is too large: auto mode selected num_blocks=" << param.num_blocks
          << ", which exceeds the maximum of " << param.project_dim << ". "
          << "Lower cos_dist or specify ham_dist/num_blocks/num_chunks manually.";
    } else {
      msg << "num_blocks must be <= " << param.project_dim << ", got " << param.num_blocks;
    }
    throw std::invalid_argument(msg.str());
  }
  if (param.ham_dist >= param.num_blocks) {
    std::ostringstream msg;
    msg << "ham_dist (" << param.ham_dist << ") must be less than num_blocks ("
        << param.num_blocks << ")";
    throw std::invalid_argument(msg.str());
  }
  if (!(param.cos_dist >= 0.f && param.cos_dist <= 2.f)) {
    std::ostringstream msg;
    msg << "cos_dist must be in [0, 2], got " << param.cos_dist;
    throw std::invalid_argument(msg.str());
  }

  if (param.verbose) std::cout << "missing edge ratio:" << calc_missing_edge_ratio(param) << std::endl;

  if (param.centering) {
    if (param.verbose) std::cerr << "start making input-data centered at 0" << std::endl;
    double centering_start = clock();
    center_data();
    double centering_end = clock();
    if (param.verbose) {
      std::cerr << "end making input-data centered at 0" << std::endl;
      std::cout << "centering time:" << (centering_end - centering_start)/(double)CLOCKS_PER_SEC << std::endl;
    }
  }

  double total_start = clock();
  precompute_norms(param.centering);

  param.radix_counts = new unsigned int[alphabet_size_];

  if (param.verbose) {
    std::cout << "number of data:" << feature_vectors_.size() << std::endl;
    std::cout << "data dimension:" << dim_ << std::endl;
    std::cout << "projected dimension:" << param.project_dim << std::endl;
    std::cout << "length of strings:" << param.project_dim * param.num_chunks << std::endl;
    std::cout << "number of chunks:" << param.num_chunks << std::endl;
  }

  double project_start = clock();
  if (param.verbose) std::cerr << "start projection" << std::endl;
  std::vector<uint8_t*> sketches;
  project_vectors(param.project_dim * param.num_chunks, sketches, param);
  if (param.verbose) std::cerr << "end projection" << std::endl;
  double project_end = clock();
  if (param.verbose) std::cout << "projecttime:" << (project_end - project_start)/(double)CLOCKS_PER_SEC << std::endl;

  param.chunk_ranges = new PosRange[param.num_chunks + 1];
  for (int i = 1; i <= (int)param.num_chunks; i++) {
    param.chunk_ranges[i].start = (int)ceil((double)param.sketch_len*((double)(i - 1)/(double)param.num_chunks)) + 1;
    param.chunk_ranges[i].end   = (int)ceil((double)param.sketch_len*(double)i/(double)param.num_chunks);
  }

  double msm_time = 0.0;

  if (param.verbose) {
    std::cerr << "chunk distance:" << param.ham_dist << std::endl;
    std::cerr << "the number of blocks:" << param.num_blocks << std::endl;
  }
  param.block_ranges = new PosRange[param.num_blocks + 1];
  for (int i = 1; i <= (int) param.num_chunks; i++) {
    param.chunk_len     = param.chunk_ranges[i].end - param.chunk_ranges[i].start + 1;
    param.chunk_start   = param.chunk_ranges[i].start;
    param.chunk_end     = param.chunk_ranges[i].end;
    param.current_chunk = i;
    for (int j = 1; j <= (int)param.num_blocks; j++) {
      param.block_ranges[j].start = (int)ceil((double)param.chunk_len*((double)(j - 1)/(double)param.num_blocks)) + param.chunk_ranges[i].start;
      param.block_ranges[j].end   = (int)ceil((double)param.chunk_len*(double)j/(double)param.num_blocks) + param.chunk_ranges[i].start - 1;
    }
    if (param.verbose) std::cerr << "start enumeration chunk no " << i << std::endl;
    double msm_start = clock();
    multi_classification(sketches, 1, 0, param.num_sketches - 1, param);
    double msm_end   = clock();
    msm_time += (msm_end - msm_start)/(double)CLOCKS_PER_SEC;
  }
  if (param.verbose) std::cout << "msmtime:" << msm_time << std::endl;

  double total_end = clock();
  if (param.verbose) {
    std::cout << "cputime:" << (total_end - total_start)/(double)CLOCKS_PER_SEC << std::endl;
    std::cout << "numSort:" << combination(param.num_blocks, param.ham_dist) * param.num_chunks << std::endl;
    std::cout << "numHamDist:" << num_ham_dist_ << std::endl;
    std::cout << "numCosDist:" << num_cos_dist_ << std::endl;
  }

  delete sketch_pool_;
  delete[] param.radix_counts;
  delete[] param.chunk_ranges;
  delete[] param.block_ranges;
}

Params SketchSort::init_params(unsigned int num_blocks,
                               unsigned int ham_dist,
                               float        cos_dist,
                               unsigned int num_chunks,
                               bool         auto_mode,
                               float        missing_ratio,
                               bool         centering,
                               unsigned int seed,
                               bool         verbose) {
  Params param;
  param.num_blocks    = num_blocks;
  param.num_chunks    = num_chunks;
  param.ham_dist      = ham_dist;
  param.cos_dist      = cos_dist;
  param.project_dim   = kProjectDim;
  param.auto_mode     = auto_mode;
  param.missing_ratio = missing_ratio;
  param.centering     = centering;
  param.seed          = seed;
  param.verbose       = verbose;
  alphabet_size_      = 2;
  return param;
}

void SketchSort::run(const char *input_path, const char *output_path,
		  unsigned int num_blocks,
		  unsigned int ham_dist,
		  float        cos_dist,
		  unsigned int num_chunks,
		  bool         auto_mode,
		  float        missing_ratio,
		  bool         centering,
		  unsigned int seed,
		  bool         verbose)
{
  Params param = init_params(num_blocks, ham_dist, cos_dist, num_chunks,
                             auto_mode, missing_ratio, centering,
                             seed, verbose);

  if (param.verbose) std::cerr << "start reading" << std::endl;
  double read_start = clock();
  read_features(input_path);
  double read_end   = clock();
  if (param.verbose) {
    std::cerr << "end reading" << std::endl;
    std::cout << "readtime:" << (read_end - read_start)/(double)CLOCKS_PER_SEC << std::endl;
  }

  std::ofstream ofs(output_path);
  if (!ofs) {
    throw std::runtime_error(std::string("cannot open output file: ") + output_path);
  }
  param.out_stream = &ofs;

  run_core(param);

  ofs.close();
}

void SketchSort::search(const float *data, std::size_t n_rows, std::size_t n_cols,
			unsigned int num_blocks,
			unsigned int ham_dist,
			float        cos_dist,
			unsigned int num_chunks,
			bool         auto_mode,
			float        missing_ratio,
			bool         centering,
			unsigned int seed,
			bool         verbose,
			std::vector<Pair> &out)
{
  if (data == nullptr) {
    throw std::invalid_argument("data pointer is null");
  }
  if (n_rows == 0 || n_cols == 0) {
    throw std::invalid_argument("X must be non-empty");
  }
  // Extra parens around `max` defeat the Windows <windows.h> `max` macro
  // that would otherwise hijack `std::numeric_limits<...>::max()` here.
  if (n_rows > (std::numeric_limits<std::uint32_t>::max)()) {
    throw std::invalid_argument("number of rows exceeds uint32 id range");
  }

  Params param = init_params(num_blocks, ham_dist, cos_dist, num_chunks,
                             auto_mode, missing_ratio, centering,
                             seed, verbose);

  out.clear();
  param.pairs = &out;

  set_features_from_raw(data, n_rows, n_cols);

  run_core(param);
}

} // namespace sketchsort
