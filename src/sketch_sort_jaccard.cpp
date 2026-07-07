/*
 * sketch_sort_jaccard.cpp
 * Copyright (c) 2011 Yasuo Tabei
 *
 * SPDX-License-Identifier: MIT
 * See the LICENSE file in the project root for the full license text.
 */

#include "sketch_sort_jaccard.hpp"

#include "multi_sort_engine.hpp"

#include <limits>
#include <stdexcept>

namespace sketchsort {
namespace jaccard {

template<class T>
static inline T tmax(T a1, T a2) { return (a1 > a2) ? a1 : a2; }

template<class T>
static inline T tmin(T a1, T a2) { return (a1 < a2) ? a1 : a2; }

// Deterministic Fisher-Yates using mt19937 directly. std::shuffle and
// std::uniform_int_distribution are NOT reproducible across standard-library
// implementations, which would make a cross-platform golden impossible; this
// hand-rolled version depends only on mt19937 (identical everywhere). The
// slight modulo bias is irrelevant to MinHash permutation quality.
static inline void det_shuffle(std::vector<std::uint32_t> &a, std::mt19937 &rng) {
  for (std::size_t i = a.size(); i > 1; --i) {
    std::uint32_t j = static_cast<std::uint32_t>(rng() % i);
    std::swap(a[i - 1], a[j]);
  }
}

void SketchSort::read_features(const char *file_name) {
  std::ifstream ifs(file_name);
  if (!ifs)
    throw std::runtime_error(std::string("cannot open input file: ") + file_name);

  max_item_ = 0;
  int val = 0;
  std::string line;
  std::size_t line_no = 0;
  while (std::getline(ifs, line)) {
    ++line_no;
    if (line.size() == 0)
      continue;
    fvs_.resize(fvs_.size() + 1);
    std::vector<std::uint32_t> &fv = fvs_[fvs_.size() - 1];
    std::istringstream is(line);
    while (is >> val) {
      if (val < 0) {
	std::ostringstream msg;
	msg << "negative id " << val << " at line " << line_no << " in " << file_name;
	throw std::runtime_error(msg.str());
      }
      if (max_item_ < val)
	max_item_ = val;
      fv.push_back(val);
    }
    if (fv.empty()) {
      fvs_.pop_back();
      continue;
    }
    std::sort(fv.begin(), fv.end());
    fv.erase(std::unique(fv.begin(), fv.end()), fv.end());
  }

  if (fvs_.empty())
    throw std::runtime_error(std::string("no data read from ") + file_name);
}

void SketchSort::set_features_from_sets(const std::vector<std::vector<std::uint32_t> > &sets) {
  // Keep every row (including empty sets) so that the returned ids match the
  // caller's input indices exactly.
  max_item_ = 0;
  fvs_.resize(sets.size());
  for (std::size_t i = 0; i < sets.size(); ++i) {
    std::vector<std::uint32_t> &fv = fvs_[i];
    fv = sets[i];
    std::sort(fv.begin(), fv.end());
    fv.erase(std::unique(fv.begin(), fv.end()), fv.end());
    if (!fv.empty()) {
      std::uint32_t hi = fv.back();  // sorted, so the last element is the max
      if (hi > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
	throw std::invalid_argument("feature id exceeds the supported range (2^31 - 1)");
      if (static_cast<int>(hi) > max_item_)
	max_item_ = static_cast<int>(hi);
    }
  }
}

void SketchSort::project_vectors(unsigned int project_dim, std::vector<uint8_t*> &sig) {
  const std::size_t stride = project_dim + 1;
  sig_storage_.assign(fvs_.size() * stride, 0);
  sig.resize(fvs_.size());
  for (std::size_t i = 0; i < sig.size(); i++) {
    sig[i] = sig_storage_.data() + i * stride;
  }

  std::vector<std::uint32_t> ptable(max_item_ + 1);
  for (std::size_t i = 0; i < project_dim; i++) {
    for (int j = 0; j <= max_item_; j++)
      ptable[j] = j;

    det_shuffle(ptable, rng_);
    for (std::size_t j = 0; j < fvs_.size(); j++) {
      std::vector<std::uint32_t> &fv  = fvs_[j];
      std::uint32_t min_p = max_item_;
      for (std::size_t k = 0; k < fv.size(); k++) {
	if (min_p > ptable[fv[k]])
	  min_p = ptable[fv[k]];
      }
      sig[j][i+1] = min_p % 256;
    }
  }
}

inline float SketchSort::calc_jaccard_dist(unsigned int id1, unsigned int id2) {
  std::vector<std::uint32_t> &fv1 = fvs_[id1];
  std::vector<std::uint32_t> &fv2 = fvs_[id2];
  std::size_t size1 = fv1.size(), size2 = fv2.size();
  std::size_t itr1 = 0, itr2 = 0;
  std::size_t num_inter_section = 0;
  std::uint32_t buffer;

  while (itr1 < size1 && itr2 < size2) {
    buffer = fv2[itr2];
    while (itr1 < size1 && fv1[itr1] < buffer) itr1++;
    if (itr1 < size1 && fv1[itr1] == buffer)   num_inter_section++;
    itr2++;
  }
  return (1.f - float(num_inter_section)/float(size1 + size2 - num_inter_section));
}

inline float SketchSort::check_upper_bound(unsigned int id1, unsigned int id2) {
  std::size_t size1 = fvs_[id1].size(), size2 = fvs_[id2].size();
  return 1.f - float(tmin(size1, size2)) / float(tmax(size1, size2));
}

double SketchSort::mismatch_prob(float jaccard_dist) {
  // Probability that one 8-bit MinHash byte differs for a pair at distance
  // jaccard_dist (b-bit minwise collision correction).
  return (255.0/256.0)*jaccard_dist;
}

Params SketchSort::init_params(unsigned int num_blocks,
                               unsigned int ham_dist,
                               float        jaccard_dist,
                               unsigned int num_chunks,
                               bool         auto_mode,
                               float        missing_ratio,
                               unsigned int seed,
                               bool         verbose) {
  Params param;
  param.num_blocks    = num_blocks;
  param.num_chunks    = num_chunks;
  param.chunk_dist    = ham_dist;
  param.jaccard_dist  = jaccard_dist;
  param.auto_mode     = auto_mode;
  param.missing_ratio = missing_ratio;
  param.verbose       = verbose;
  rng_.seed(seed);
  return param;
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
        engine::decide_parameters(mismatch_prob(param.jaccard_dist), param.missing_ratio, kProjectDim);
    param.chunk_dist = decided.ham_dist;
    param.num_blocks = decided.num_blocks;
    param.num_chunks = decided.num_chunks;
    if (param.verbose) {
      std::cout << "decided parameters:" << std::endl;
      std::cout << "hamming distance threshold: " << param.chunk_dist << std::endl;
      std::cout << "number of blocks: " << param.num_blocks << std::endl;
      std::cout << "number of chunks: "  << param.num_chunks << std::endl;
      std::cout << std::endl;
    }
  }

  engine::validate_params(param.num_blocks, param.num_chunks, param.chunk_dist, kProjectDim,
                          param.auto_mode, param.jaccard_dist, 0.f, 1.f, "jaccard_dist");

  if (param.verbose) std::cout << "missing edge ratio:"
                               << engine::missing_edge_ratio(mismatch_prob(param.jaccard_dist), kProjectDim,
                                                             param.chunk_dist, param.num_chunks)
                               << std::endl;

  double total_start = clock();

  if (param.verbose) {
    std::cout << "number of data:" << fvs_.size() << std::endl;
    std::cout << "projected dimension:" << kProjectDim << std::endl;
    std::cout << "length of strings:" << kProjectDim * param.num_chunks << std::endl;
    std::cout << "number of chunks:" << param.num_chunks << std::endl;
  }

  double project_start = clock();
  if (param.verbose) std::cerr << "start projection" << std::endl;
  std::vector<uint8_t*> sig;
  project_vectors(kProjectDim * param.num_chunks, sig);
  if (param.verbose) std::cerr << "end projection" << std::endl;
  double project_end = clock();
  if (param.verbose) std::cout << "projecttime:" << (project_end - project_start)/(double)CLOCKS_PER_SEC << std::endl;

  if (param.verbose) {
    std::cerr << "chunk distance:" << param.chunk_dist << std::endl;
    std::cerr << "the number of blocks:" << param.num_blocks << std::endl;
  }

  // Exact-verification + emission hook handed to the shared enumeration
  // engine. The set-size ratio gives an O(1) lower bound on the Jaccard
  // distance, used as a prefilter before any sketch comparison; reported
  // pairs are normalized to id1 < id2.
  struct Emitter {
    SketchSort *self;
    Params *p;
    bool prefilter(unsigned int id1, unsigned int id2) {
      return self->check_upper_bound(id1, id2) <= p->jaccard_dist;
    }
    void candidate(unsigned int id1, unsigned int id2) {
      float jaccard_dist = self->calc_jaccard_dist(id1, id2);
      if (jaccard_dist <= p->jaccard_dist) {
	if (id1 > id2) { unsigned int t = id1; id1 = id2; id2 = t; }
	if (p->pairs) {
	  Pair pr;
	  pr.id1          = static_cast<std::uint32_t>(id1);
	  pr.id2          = static_cast<std::uint32_t>(id2);
	  pr.jaccard_dist = jaccard_dist;
	  p->pairs->push_back(pr);
	}
	if (p->os) {
	  (*p->os) << id1 << " " << id2 << " " << jaccard_dist << "\n";
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
}

void SketchSort::run(const char *file_name, const char *output_name,
		     unsigned int num_blocks,
		     unsigned int ham_dist,
		     float        jaccard_dist,
		     unsigned int num_chunks,
		     bool         auto_mode,
		     float        missing_ratio,
		     unsigned int seed,
		     bool         verbose)
{
  Params param = init_params(num_blocks, ham_dist, jaccard_dist, num_chunks,
                             auto_mode, missing_ratio, seed, verbose);

  if (param.verbose) std::cerr << "start reading" << std::endl;
  double read_start = clock();
  read_features(file_name);
  double read_end   = clock();
  if (param.verbose) {
    std::cerr << "end reading" << std::endl;
    std::cout << "readtime:" << (read_end - read_start)/(double)CLOCKS_PER_SEC << std::endl;
  }

  std::ofstream ofs(output_name);
  if (!ofs)
    throw std::runtime_error(std::string("cannot open output file: ") + output_name);
  param.os = &ofs;

  run_core(param);

  ofs.close();
}

void SketchSort::search(const std::vector<std::vector<std::uint32_t> > &sets,
			unsigned int num_blocks,
			unsigned int ham_dist,
			float        jaccard_dist,
			unsigned int num_chunks,
			bool         auto_mode,
			float        missing_ratio,
			unsigned int seed,
			bool         verbose,
			std::vector<Pair> &out)
{
  if (sets.empty())
    throw std::invalid_argument("input set collection must be non-empty");
  if (sets.size() > (std::numeric_limits<std::uint32_t>::max)())
    throw std::invalid_argument("number of sets exceeds uint32 id range");

  Params param = init_params(num_blocks, ham_dist, jaccard_dist, num_chunks,
                             auto_mode, missing_ratio, seed, verbose);

  out.clear();
  param.pairs = &out;

  set_features_from_sets(sets);

  run_core(param);
}

}  // namespace jaccard
}  // namespace sketchsort
