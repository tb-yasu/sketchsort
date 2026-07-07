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

#include "multi_sort_engine.hpp"

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

void SketchSort::project_vectors(unsigned int project_dim, std::vector<uint8_t*> &sketches, unsigned int seed) {
  std::vector<float> rand_vec;
  sketch_pool_ = new boost::pool<>(sizeof(uint8_t));
  sketches.resize(feature_vectors_.size());
  for (size_t i = 0; i < sketches.size(); i++) {
    sketches[i]    = (uint8_t*)sketch_pool_->ordered_malloc(project_dim + 1);
  }

  boost::mt19937 gen(static_cast<unsigned long>(seed));
  boost::normal_distribution<> normal_dist(0.f, 1.f);
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > rand(gen, normal_dist);

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
}

inline float SketchSort::calc_cos_dist(unsigned int id1, unsigned int id2) {
  boost::numeric::ublas::vector<float> &vec1 = feature_vectors_[id1];
  boost::numeric::ublas::vector<float> &vec2 = feature_vectors_[id2];
  float dot = boost::numeric::ublas::inner_prod(vec1, vec2);

  return (1.f - dot*(inv_norms_[id1]*inv_norms_[id2]));
}

double SketchSort::mismatch_prob(float cos_dist) {
  // Probability that one sign bit differs for a pair at distance cos_dist.
  return acos(1.0 - cos_dist)/M_PI;
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

void SketchSort::run_core(Params &param) {
  if (param.auto_mode) {
    if (!(param.missing_ratio > 0.f && param.missing_ratio < 1.f)) {
      std::ostringstream msg;
      msg << "missing_ratio must be in (0, 1), got " << param.missing_ratio;
      throw std::invalid_argument(msg.str());
    }
    if (param.verbose) std::cerr << "deciding parameters such that the missing edge ratio is no more than " << param.missing_ratio << std::endl;
    engine::AutoParams decided =
        engine::decide_parameters(mismatch_prob(param.cos_dist), param.missing_ratio, kProjectDim);
    param.ham_dist   = decided.ham_dist;
    param.num_blocks = decided.num_blocks;
    param.num_chunks = decided.num_chunks;
    if (param.verbose) {
      std::cout << "decided parameters:" << std::endl;
      std::cout << "hamming distance threshold: " << param.ham_dist << std::endl;
      std::cout << "number of blocks: " << param.num_blocks << std::endl;
      std::cout << "number of chunks: "  << param.num_chunks << std::endl;
      std::cout << std::endl;
    }
  }

  engine::validate_params(param.num_blocks, param.num_chunks, param.ham_dist, kProjectDim,
                          param.auto_mode, param.cos_dist, 0.f, 2.f, "cos_dist");

  if (param.verbose) std::cout << "missing edge ratio:"
                               << engine::missing_edge_ratio(mismatch_prob(param.cos_dist), kProjectDim,
                                                             param.ham_dist, param.num_chunks)
                               << std::endl;

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

  if (param.verbose) {
    std::cout << "number of data:" << feature_vectors_.size() << std::endl;
    std::cout << "data dimension:" << dim_ << std::endl;
    std::cout << "projected dimension:" << kProjectDim << std::endl;
    std::cout << "length of strings:" << kProjectDim * param.num_chunks << std::endl;
    std::cout << "number of chunks:" << param.num_chunks << std::endl;
  }

  double project_start = clock();
  if (param.verbose) std::cerr << "start projection" << std::endl;
  std::vector<uint8_t*> sketches;
  project_vectors(kProjectDim * param.num_chunks, sketches, param.seed);
  if (param.verbose) std::cerr << "end projection" << std::endl;
  double project_end = clock();
  if (param.verbose) std::cout << "projecttime:" << (project_end - project_start)/(double)CLOCKS_PER_SEC << std::endl;

  if (param.verbose) {
    std::cerr << "chunk distance:" << param.ham_dist << std::endl;
    std::cerr << "the number of blocks:" << param.num_blocks << std::endl;
  }

  // Exact-verification + emission hook handed to the shared enumeration
  // engine; sign sketches use a two-letter alphabet.
  struct Emitter {
    SketchSort *self;
    Params *p;
    bool prefilter(unsigned int, unsigned int) { return true; }
    void candidate(unsigned int id1, unsigned int id2) {
      float cos_dist = self->calc_cos_dist(id1, id2);
      if (cos_dist <= p->cos_dist) {
	if (p->pairs) {
	  Pair pr;
	  pr.id1      = static_cast<std::uint32_t>(id1);
	  pr.id2      = static_cast<std::uint32_t>(id2);
	  pr.cos_dist = cos_dist;
	  p->pairs->push_back(pr);
	}
	if (p->out_stream) {
	  (*p->out_stream) << id1 << " " << id2 << " " << cos_dist << "\n";
	}
      }
    }
  } emitter{this, &param};

  engine::Config cfg;
  cfg.num_blocks    = param.num_blocks;
  cfg.num_chunks    = param.num_chunks;
  cfg.ham_dist      = param.ham_dist;
  cfg.alphabet_size = 2;
  cfg.sketch_len    = kProjectDim * param.num_chunks;
  cfg.verbose       = param.verbose;
  engine::MultiSortEngine<Emitter> eng(cfg, sketches, emitter);

  double msm_start = clock();
  eng.run();
  double msm_end   = clock();
  if (param.verbose) std::cout << "msmtime:" << (msm_end - msm_start)/(double)CLOCKS_PER_SEC << std::endl;

  double total_end = clock();
  if (param.verbose) {
    std::cout << "cputime:" << (total_end - total_start)/(double)CLOCKS_PER_SEC << std::endl;
    std::cout << "numSort:" << engine::combination(param.num_blocks, param.ham_dist) * param.num_chunks << std::endl;
    std::cout << "numHamDist:" << eng.stats().ham_checks << std::endl;
    std::cout << "numCosDist:" << eng.stats().candidates << std::endl;
  }

  delete sketch_pool_;
  sketch_pool_ = nullptr;
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
  param.auto_mode     = auto_mode;
  param.missing_ratio = missing_ratio;
  param.centering     = centering;
  param.seed          = seed;
  param.verbose       = verbose;
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
