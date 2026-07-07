/*
 * sketch_sort_minmax.cpp
 * Copyright (c) 2017 Yasuo Tabei
 *
 * SPDX-License-Identifier: MIT
 * See the LICENSE file in the project root for the full license text.
 */

#include "sketch_sort_minmax.hpp"

#include "multi_sort_engine.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace sketchsort {
namespace minmax {

float SketchSort::safe_log(float val) {
  if (val < 1.0e-20f)
    return -1.0e+20f;
  return std::log(val);
}

void SketchSort::read_features(const char *input_path) {
  std::ifstream ifs(input_path);
  if (!ifs)
    throw std::runtime_error(std::string("cannot open input file: ") + input_path);

  std::string line;
  while (std::getline(ifs, line)) {
    if (line.size() == 0)
      continue;
    fvs_.emplace_back();
    std::vector<float> &fv = fvs_.back();

    const char *p = line.c_str();
    char *end = nullptr;
    if (fvs_.size() == 1) {
      for (float val = strtof(p, &end); end != p; val = strtof(p, &end)) {
	fv.push_back(val);
	p = end;
      }
      dim_ = fv.size();
      if (dim_ == 0)
	throw std::runtime_error("the first line contains no numbers");
    }
    else {
      fv.reserve(dim_);
      for (float val = strtof(p, &end); end != p; val = strtof(p, &end)) {
	if (fv.size() >= dim_)
	  throw std::runtime_error("input rows have inconsistent dimensions");
	fv.push_back(val);
	p = end;
      }
      if (fv.size() != dim_)
	throw std::runtime_error("input rows have inconsistent dimensions");
    }
  }

  if (fvs_.empty())
    throw std::runtime_error(std::string("no data read from ") + input_path);
}

void SketchSort::set_features_from_raw(const float *data, std::size_t n_rows, std::size_t n_cols) {
  dim_ = n_cols;
  fvs_.resize(n_rows);
  for (std::size_t i = 0; i < n_rows; ++i) {
    fvs_[i].assign(data + i * n_cols, data + (i + 1) * n_cols);
  }
}

void SketchSort::generate_matrix(std::uint32_t project_dim, unsigned long seed,
                                 std::vector<std::vector<float> > &mat_r,
                                 std::vector<std::vector<float> > &mat_c,
                                 std::vector<std::vector<float> > &mat_b) {
  std::uint64_t dat_dim2 = dim_ << 1;

  // A single generator is shared across the three matrices so that R, C and B
  // are drawn from independent parts of the same random stream. Using one
  // generator per matrix seeded with time(0) would make them identical.
  boost::mt19937 gen(seed);
  boost::gamma_distribution<>  gdst(2.0);
  boost::uniform_real<>        udst(0, 1.0);
  boost::variate_generator<boost::mt19937&, boost::gamma_distribution<> > grand(gen, gdst);
  boost::variate_generator<boost::mt19937&, boost::uniform_real<> >       urand(gen, udst);

  mat_r.resize(project_dim);
  for (std::size_t i = 0; i < project_dim; ++i) {
    std::vector<float> &vec_r = mat_r[i];
    vec_r.resize(dat_dim2);
    for (std::size_t j = 0; j < dat_dim2; ++j)
      vec_r[j] = grand();
  }

  mat_c.resize(project_dim);
  for (std::size_t i = 0; i < project_dim; ++i) {
    std::vector<float> &vec_c = mat_c[i];
    vec_c.resize(dat_dim2);
    for (std::size_t j = 0; j < dat_dim2; ++j)
      vec_c[j] = grand();
  }

  mat_b.resize(project_dim);
  for (std::size_t i = 0; i < project_dim; ++i) {
    std::vector<float> &vec_b = mat_b[i];
    vec_b.resize(dat_dim2);
    for (std::size_t j = 0; j < dat_dim2; ++j)
      vec_b[j] = urand();
  }
}

void SketchSort::project_vectors(std::uint32_t project_dim, std::vector<uint8_t*> &sig,
                                 unsigned long seed, bool verbose) {
  std::vector<std::vector<float> > mat_r;
  std::vector<std::vector<float> > mat_c;
  std::vector<std::vector<float> > mat_b;
  generate_matrix(project_dim, seed, mat_r, mat_c, mat_b);

  std::uint64_t dat_dim2 = dim_ << 1;

  // Precompute log(C) and 1/R once so the hot loop below has no divisions and
  // no log() of the projection matrices.
  std::vector<std::vector<float> > log_c(project_dim);
  std::vector<std::vector<float> > inv_r(project_dim);
  for (std::size_t j = 0; j < project_dim; ++j) {
    log_c[j].resize(dat_dim2);
    inv_r[j].resize(dat_dim2);
    for (std::size_t e = 0; e < dat_dim2; ++e) {
      log_c[j][e] = safe_log(mat_c[j][e]);
      inv_r[j][e] = 1.0f / mat_r[j][e];
    }
  }

  std::size_t row_len = (std::size_t)project_dim + 1;
  sig_pool_ = new uint8_t[fvs_.size() * row_len];
  sig.resize(fvs_.size());
  for (std::size_t i = 0; i < sig.size(); i++) {
    sig[i] = sig_pool_ + i * row_len;
  }

  // Reused per data point: the nonzero elements of the expanded 2*dim_ vector.
  // Only one of the (positive, negative) parts of each coordinate is nonzero,
  // and a zero weight can never be the argmin, so zeros are skipped entirely.
  std::vector<std::uint64_t> nz_ind;
  std::vector<float>         nz_log;
  nz_ind.reserve(dat_dim2);
  nz_log.reserve(dat_dim2);

  for (std::size_t i = 0; i < fvs_.size(); ++i) {
    if (verbose && i % 1000 == 0)
      std::cerr << "project " << i << " < " << fvs_.size() << std::endl;

    std::vector<float> &fv = fvs_[i];

    nz_ind.clear();
    nz_log.clear();
    for (std::size_t k = 0; k < dim_; ++k) {
      float v  = fv[k];
      float av = std::fabs(v);
      if (av < 1.0e-20f)
	continue;
      nz_ind.push_back((v > 0) ? (k << 1) : ((k << 1) + 1));
      nz_log.push_back(std::log(av));
    }

    for (std::size_t j = 0; j < project_dim; ++j) {
      const float *vec_r     = mat_r[j].data();
      const float *vec_b     = mat_b[j].data();
      const float *vec_log_c = log_c[j].data();
      const float *vec_inv_r = inv_r[j].data();

      float min_val = 3.402823466e+35F;
      std::uint64_t min_index = 0;
      for (std::size_t m = 0, n = nz_ind.size(); m < n; ++m) {
	std::uint64_t ind = nz_ind[m];
	std::int64_t t = (std::int64_t)std::floor(nz_log[m] * vec_inv_r[ind] + vec_b[ind]);
	float a = vec_log_c[ind] - vec_r[ind] * ((float)t + 1.0f - vec_b[ind]);
	if (a < min_val) {
	  min_val = a;
	  min_index = ind;
	}
      }
      sig[i][j+1] = min_index % 256;
    }
  }
}

inline float SketchSort::calc_minmax_dist(std::uint32_t id1, std::uint32_t id2) {
  std::vector<float> &fv1 = fvs_[id1];
  std::vector<float> &fv2 = fvs_[id2];

  double num = 0.0, den = 0.0;
  for (std::size_t d = 0; d < dim_; ++d) {
    if (fv1[d] > 0 && fv2[d] > 0) {
      num += std::min(fv1[d], fv2[d]);
      den += std::max(fv1[d], fv2[d]);
    }
    else if (fv1[d] < 0 && fv2[d] < 0) {
      num += std::min(-fv1[d], -fv2[d]);
      den += std::max(-fv1[d], -fv2[d]);
    }
    else {
      den += std::fabs(fv1[d]);
      den += std::fabs(fv2[d]);
    }
  }

  if (den < 1.0e-30)  // both vectors are all-zero: treat as maximally distant
    return 1.f;

  return (1.f - num/den);
}

double SketchSort::mismatch_prob(float minmax_dist) {
  // Probability that one 8-bit GCWS hash byte differs for a pair at distance
  // minmax_dist (b-bit minwise collision correction).
  return (255.0/256.0)*minmax_dist;
}

void SketchSort::z_normalize() {
  std::size_t num = fvs_.size();
  if (num < 2)  // variance is undefined for a single vector
    return;
  for (std::size_t d = 0; d < dim_; ++d) {
    double ave = 0.0;
    for (std::size_t i = 0; i < num; ++i)
      ave += fvs_[i][d];
    ave /= (double)num;

    double sigma = 0.0;
    for (std::size_t i = 0; i < num; ++i)
      sigma += (fvs_[i][d] - ave) * (fvs_[i][d] - ave);
    sigma /= (num - 1);
    sigma = std::sqrt(sigma);
    if (sigma < 1.0e-30)
      continue;
    for (std::size_t i = 0; i < num; ++i)
      fvs_[i][d] = (fvs_[i][d] - ave)/sigma;
  }
}

void SketchSort::minmax_normalize() {
  std::size_t num = fvs_.size();
  for (std::size_t d = 0; d < dim_; ++d) {
    double min_val = DBL_MAX, max_val = -DBL_MAX;
    for (std::size_t i = 0; i < num; ++i) {
      if (fvs_[i][d] > max_val)
	max_val = fvs_[i][d];
      if (fvs_[i][d] < min_val)
	min_val = fvs_[i][d];
    }
    if (max_val - min_val < 1.0e-30)  // constant column: leave it unchanged
      continue;
    for (std::size_t i = 0; i < num; ++i)
      fvs_[i][d] = (fvs_[i][d] - min_val)/(max_val - min_val);
  }
}

Params SketchSort::init_params(std::uint32_t num_blocks,
                               std::uint32_t ham_dist,
                               float         minmax_dist,
                               std::uint32_t num_chunks,
                               bool          auto_mode,
                               float         missing_ratio,
                               bool          z_normalization,
                               bool          minmax_normalization,
                               long          seed,
                               bool          verbose) {
  Params param;
  param.num_blocks           = num_blocks;
  param.num_chunks           = num_chunks;
  param.chunk_dist           = ham_dist;
  param.minmax_dist          = minmax_dist;
  param.auto_mode            = auto_mode;
  param.missing_ratio        = missing_ratio;
  param.z_normalization      = z_normalization;
  param.minmax_normalization = minmax_normalization;
  param.verbose              = verbose;
  param.seed = (seed < 0) ? static_cast<unsigned long>(time(nullptr))
                          : static_cast<unsigned long>(seed);
  return param;
}

void SketchSort::run_core(Params &param) {
  if (param.verbose) std::cout << "random seed: " << param.seed << std::endl;

  if (param.auto_mode) {
    if (!(param.missing_ratio > 0.f && param.missing_ratio < 1.f)) {
      std::ostringstream msg;
      msg << "missing_ratio must be in (0, 1), got " << param.missing_ratio;
      throw std::invalid_argument(msg.str());
    }
    if (param.verbose) std::cerr << "tune parameters such that the missing edge ratio is no more than " << param.missing_ratio << std::endl;
    engine::AutoParams decided =
        engine::decide_parameters(mismatch_prob(param.minmax_dist), param.missing_ratio, kProjectDim);
    param.chunk_dist = decided.ham_dist;
    param.num_blocks = decided.num_blocks;
    param.num_chunks = decided.num_chunks;
    if (param.verbose) {
      std::cout << "tuned parameters:" << std::endl;
      std::cout << "hamming distance threshold: " << param.chunk_dist << std::endl;
      std::cout << "number of blocks: " << param.num_blocks << std::endl;
      std::cout << "number of chunks: "  << param.num_chunks << std::endl;
      std::cout << std::endl;
    }
  }

  engine::validate_params(param.num_blocks, param.num_chunks, param.chunk_dist, kProjectDim,
                          param.auto_mode, param.minmax_dist, 0.f, 1.f, "minmax_dist");

  if (param.verbose) std::cout << "missing edge ratio:"
                               << engine::missing_edge_ratio(mismatch_prob(param.minmax_dist), kProjectDim,
                                                             param.chunk_dist, param.num_chunks)
                               << std::endl;

  double total_start = clock();

  if (param.z_normalization) {
    if (param.verbose) std::cout << "z-normalizing" << std::endl;
    z_normalize();
  }
  else if (param.minmax_normalization) {
    if (param.verbose) std::cout << "minmax normalizing" << std::endl;
    minmax_normalize();
  }

  if (param.verbose) {
    std::cout << "number of data:" << fvs_.size() << std::endl;
    std::cout << "length of strings per chunk:" << kProjectDim << std::endl;
    std::cout << "number of chunks:" << param.num_chunks << std::endl;
    std::cout << "total length of strings:" << kProjectDim * param.num_chunks << std::endl;
  }

  double project_start = clock();
  if (param.verbose) std::cerr << "start projection" << std::endl;
  std::vector<uint8_t*> sig;
  project_vectors(kProjectDim * param.num_chunks, sig, param.seed, param.verbose);
  if (param.verbose) std::cerr << "end projection" << std::endl;
  double project_end = clock();
  if (param.verbose) std::cout << "projecttime:" << (project_end - project_start)/(double)CLOCKS_PER_SEC << std::endl;

  if (param.verbose) {
    std::cerr << "chunk distance:" << param.chunk_dist << std::endl;
    std::cerr << "the number of blocks:" << param.num_blocks << std::endl;
  }

  // Exact-verification + emission hook handed to the shared enumeration
  // engine; GCWS hash bytes use the full 256-letter alphabet.
  struct Emitter {
    SketchSort *self;
    Params *p;
    bool prefilter(std::uint32_t, std::uint32_t) { return true; }
    void candidate(std::uint32_t id1, std::uint32_t id2) {
      float dist = self->calc_minmax_dist(id1, id2);
      if (dist <= p->minmax_dist) {
	if (p->pairs) {
	  Pair pr;
	  pr.id1         = id1;
	  pr.id2         = id2;
	  pr.minmax_dist = dist;
	  p->pairs->push_back(pr);
	}
	if (p->os) {
	  (*p->os) << id1 << " " << id2 << " " << dist << "\n";
	}
      }
    }
  } emitter{this, &param};

  engine::Config cfg;
  cfg.num_blocks    = param.num_blocks;
  cfg.num_chunks    = param.num_chunks;
  cfg.ham_dist      = param.chunk_dist;
  cfg.alphabet_size = 256;
  cfg.sketch_len    = kProjectDim * param.num_chunks;
  cfg.verbose       = param.verbose;
  engine::MultiSortEngine<Emitter> eng(cfg, sig, emitter);

  double msm_start = clock();
  eng.run();
  double msm_end   = clock();
  if (param.verbose) std::cout << "msmtime:" << (msm_end - msm_start)/(double)CLOCKS_PER_SEC << std::endl;

  double total_end = clock();
  if (param.verbose) std::cout << "cputime:" << (total_end - total_start)/(double)CLOCKS_PER_SEC << std::endl;

  delete[] sig_pool_;
  sig_pool_ = nullptr;
}

void SketchSort::run(const char *input_path, const char *output_path,
		     std::uint32_t num_blocks,
		     std::uint32_t ham_dist,
		     float         minmax_dist,
		     std::uint32_t num_chunks,
		     bool          auto_mode,
		     float         missing_ratio,
		     bool          z_normalization,
		     bool          minmax_normalization,
		     long          seed,
		     bool          verbose)
{
  Params param = init_params(num_blocks, ham_dist, minmax_dist, num_chunks,
                             auto_mode, missing_ratio, z_normalization,
                             minmax_normalization, seed, verbose);

  if (param.verbose) std::cerr << "start reading" << std::endl;
  double read_start = clock();
  read_features(input_path);
  double read_end   = clock();
  if (param.verbose) {
    std::cerr << "end reading" << std::endl;
    std::cout << "readtime:" << (read_end - read_start)/(double)CLOCKS_PER_SEC << std::endl;
  }

  std::ofstream ofs(output_path);
  if (!ofs)
    throw std::runtime_error(std::string("cannot open output file: ") + output_path);
  param.os = &ofs;

  run_core(param);

  ofs.close();
}

void SketchSort::search(const float *data, std::size_t n_rows, std::size_t n_cols,
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
			std::vector<Pair> &out)
{
  if (data == nullptr)
    throw std::invalid_argument("data pointer is null");
  if (n_rows == 0 || n_cols == 0)
    throw std::invalid_argument("X must be non-empty");
  // Extra parens defeat the Windows <windows.h> max() macro.
  if (n_rows > (std::numeric_limits<std::uint32_t>::max)())
    throw std::invalid_argument("number of rows exceeds uint32 id range");

  Params param = init_params(num_blocks, ham_dist, minmax_dist, num_chunks,
                             auto_mode, missing_ratio, z_normalization,
                             minmax_normalization, seed, verbose);

  out.clear();
  param.pairs = &out;

  set_features_from_raw(data, n_rows, n_cols);

  run_core(param);
}

}  // namespace minmax
}  // namespace sketchsort
