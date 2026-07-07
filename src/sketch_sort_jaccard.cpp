/*
 * sketch_sort_jaccard.cpp
 * Copyright (c) 2011 Yasuo Tabei
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * See the LICENSE file in the project root for the full license text.
 */

#include "sketch_sort_jaccard.hpp"

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

int SketchSort::project_vectors(unsigned int project_dim, std::vector<uint8_t*> &sig, Params &param) {
  const std::size_t stride = project_dim + 1;
  sig_storage_.assign(fvs_.size() * stride, 0);
  sig.resize(fvs_.size());
  param.ids.resize(fvs_.size());
  for (std::size_t i = 0; i < sig.size(); i++) {
    sig[i]       = sig_storage_.data() + i * stride;
    param.ids[i] = i;
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
  param.seq_len = project_dim;
  param.num_seq = fvs_.size();

  return 1;
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

inline void SketchSort::radix_sort(std::vector<uint8_t*> &sig, int spos, int epos, int l, int r, Params &param) {
  unsigned int *c      = param.counter;
  std::vector<unsigned int> &ids      = param.ids;
  std::vector<uint8_t*> new_sig(r - l + 1);
  std::vector<unsigned int> new_ids(r - l + 1);
  unsigned int tmp;
  int tpos = spos - 1;
  while (++tpos <= epos) {
    for (unsigned int i = 0; i < num_char_; i++) *(c + i) = 0;
    for (int i = l; i <= r; i++) c[sig[i][tpos]]++;
    for (unsigned int i = 1; i < num_char_; i++) *(c + i) += *(c + i - 1);
    for (int i = r; i >= l; --i) {
      tmp = --c[sig[i][tpos]] + l;
      new_ids[tmp - l] = ids[i];
      new_sig[tmp - l] = sig[i];
    }
    if (++tpos <= epos) {
      for (unsigned int i = 0; i < num_char_; i++) *(c + i) = 0;
      for (int i = l; i <= r; i++) c[new_sig[i - l][tpos]]++;
      for (unsigned int i = 1; i < num_char_; i++) *(c + i) += *(c + i - 1);
      for (int i = r; i >= l; --i) {
	tmp = --c[new_sig[i - l][tpos]] + l;
	ids[tmp] = new_ids[i - l];
	sig[tmp] = new_sig[i - l];
      }
    }
    else {
      for (int i = l; i <= r; i++) {
	ids[i] = new_ids[i - l];
	sig[i] = new_sig[i - l];
      }
      return;
    }
  }
}

inline void SketchSort::insertion_sort(std::vector<uint8_t*> &sig, int spos, int epos, int l, int r, Params &param) {
  int i, j;
  uint8_t *pivot, pval;
  unsigned int pid;
  std::vector<unsigned int> &ids = param.ids;
  for (int tpos = spos; tpos <= epos; tpos++) {
    for (i = l + 1; i <= r; i++) {
      pivot = sig[i]; pval = sig[i][tpos]; pid = ids[i];
      for (j = i; j > l && sig[j-1][tpos] > pval; j--) {
	sig[j] = sig[j-1];
	ids[j] = ids[j-1];
      }
      sig[j] = pivot;
      ids[j] = pid;
    }
  }
}

inline void SketchSort::classify(std::vector<uint8_t*> &sig, int spos, int epos, int l, int r, int bpos, Params &param) {
  int n_l = l, n_r = r;
  for (int iter = l + 1; iter <= r; iter++) {
    if (!std::equal(sig[n_l] + spos, sig[n_l] + epos + 1, sig[iter] + spos)) {
      n_r = iter - 1;
      if (n_r - n_l >= 1)
	multi_classification(sig, bpos + 1, n_l, n_r, param);
      n_l = iter;
    }
  }
  if (r - n_l >= 1)
    multi_classification(sig, bpos + 1, n_l, r, param);
}

inline bool SketchSort::calc_chunk_hamdist(uint8_t *seq1, uint8_t *seq2, const Params &param) {
  unsigned int d = 0;
  for (std::size_t i = 1;  i <= param.chunk_len; i++)
    if (*seq1++ != *seq2++ && ++d > param.chunk_dist) return false;
  return true;
}

inline bool SketchSort::check_chunk_canonical(uint8_t *seq1, uint8_t *seq2, const Params &param) {
  unsigned int d = 0;
  int end        = param.pchunks[param.cchunk].start - 1;
  int j          = 1;
  int tend       = param.pchunks[j].end;
  int i          = 0;

  while (++i <= end) {
    if (seq1[i] != seq2[i] && ++d > param.chunk_dist) {
      while (++i <= tend)
	if (seq1[i] != seq2[i]) ++d;
      d = 0;
      tend = param.pchunks[++j].end;
      i    = param.pchunks[j].start - 1;
      continue;
    }
    if (tend == i)
      return false;
  }
  return true;
}

inline bool SketchSort::check_canonical(uint8_t *seq1, uint8_t *seq2, const Params &param) {
  std::size_t sb = 1, eb = 1;
  std::size_t b;
  for (std::size_t i = 0, size = param.blocks.size(); i < size; i++) {
    eb = param.blocks[i];
    for (b = sb; b < eb; b++) {
      if (std::equal(seq1 + param.pos[b].start, seq1 + param.pos[b].end + 1, seq2 + param.pos[b].start))
	  return false;
    }
    sb = param.blocks[i] + 1;
  }
  return true;
}

inline void SketchSort::report(std::vector<uint8_t*> &sig, int l, int r, Params &param) {
  float jaccard_dist;
  for (int i = l; i < r; i++) {
    for (int j = i + 1; j <= r; j++) {
      if (check_upper_bound(param.ids[i], param.ids[j]) <= param.jaccard_dist &&
	  calc_chunk_hamdist(sig[i] + param.start_chunk, sig[j] + param.start_chunk, param) &&
	  check_canonical(sig[i], sig[j], param) &&
	  check_chunk_canonical(sig[i], sig[j], param) &&
	  ((jaccard_dist = calc_jaccard_dist(param.ids[i], param.ids[j])) <= param.jaccard_dist)) {
	unsigned int id1 = param.ids[i], id2 = param.ids[j];
	if (id1 > id2) { unsigned int t = id1; id1 = id2; id2 = t; }
	if (param.pairs) {
	  Pair pr;
	  pr.id1          = static_cast<std::uint32_t>(id1);
	  pr.id2          = static_cast<std::uint32_t>(id2);
	  pr.jaccard_dist = jaccard_dist;
	  param.pairs->push_back(pr);
	}
	if (param.os) {
	  (*param.os) << id1 << " " << id2 << " " << jaccard_dist << "\n";
	}
      }
    }
  }
}

void SketchSort::multi_classification(std::vector<uint8_t*> &sig, int maxind, int l, int r, Params &param) {
  if (param.blocks.size() == param.num_blocks - param.chunk_dist) {
    report(sig, l, r, param);
    return;
  }

  for (int bpos = maxind; bpos <= (int)param.num_blocks; bpos++) {
    if (param.blocks.size() + (param.num_blocks - bpos + 1) < param.num_blocks - param.chunk_dist) { // pruning
      return;
    }
    param.blocks.push_back(bpos);
    radix_sort(sig, param.pos[bpos].start, param.pos[bpos].end, l, r, param);
    classify(sig, param.pos[bpos].start, param.pos[bpos].end, l, r, bpos, param);
    param.blocks.pop_back();
  }
}

static double combination(int n, int m) {
  double sum = 1.0;
  for (int i = 0; i < m; i++) {
    sum *= (static_cast<double>(n - i) / static_cast<double>(m - i));
  }
  return sum;
}

double SketchSort::calc_missing_edge_ratio(Params &param) {
  double sum = 0.f;
  double prob = (255.0/256.0)*param.jaccard_dist;
  unsigned int l = param.project_dim;
  unsigned int d = param.chunk_dist;
  for (unsigned int k = 0; k <= d; k++) {
    sum += (combination(l, k) * pow(prob, k) * pow(1.f - prob, l - k));
  }
  return pow(1.f - sum, param.num_chunks);
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
    param.chunk_dist = ham_dist;
    param.num_blocks = num_blocks;
    param.num_chunks = num_chunks;
  } while (calc_missing_edge_ratio(param) >= missing_ratio);
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
  param.project_dim   = kProjectDim;
  param.auto_mode     = auto_mode;
  param.missing_ratio = missing_ratio;
  param.verbose       = verbose;
  rng_.seed(seed);
  num_char_ = 256;
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
    decide_parameters(param.missing_ratio, param);
    if (param.verbose) {
      std::cout << "decided parameters:" << std::endl;
      std::cout << "hamming distance threshold: " << param.chunk_dist << std::endl;
      std::cout << "number of blocks: " << param.num_blocks << std::endl;
      std::cout << "number of chunks: "  << param.num_chunks << std::endl;
      std::cout << std::endl;
    }
  }

  if (param.num_chunks < 1)
    throw std::invalid_argument("num_chunks must be >= 1");
  if (param.num_blocks < 1) {
    std::ostringstream msg;
    msg << "num_blocks must be >= 1, got " << param.num_blocks;
    throw std::invalid_argument(msg.str());
  }
  if (param.num_blocks > param.project_dim) {
    std::ostringstream msg;
    msg << "num_blocks (" << param.num_blocks << ") cannot exceed the chunk length ("
        << param.project_dim << ")";
    throw std::invalid_argument(msg.str());
  }
  if (param.chunk_dist >= param.num_blocks) {
    std::ostringstream msg;
    msg << "ham_dist (" << param.chunk_dist << ") must be less than num_blocks ("
        << param.num_blocks << ")";
    throw std::invalid_argument(msg.str());
  }
  if (!(param.jaccard_dist >= 0.f && param.jaccard_dist <= 1.f)) {
    std::ostringstream msg;
    msg << "jaccard_dist must be in [0, 1], got " << param.jaccard_dist;
    throw std::invalid_argument(msg.str());
  }

  if (param.verbose) std::cout << "missing edge ratio:" << calc_missing_edge_ratio(param) << std::endl;

  double total_start = clock();

  param.counter = new unsigned int[num_char_];

  if (param.verbose) {
    std::cout << "number of data:" << fvs_.size() << std::endl;
    std::cout << "projected dimension:" << param.project_dim << std::endl;
    std::cout << "length of strings:" << param.project_dim * param.num_chunks << std::endl;
    std::cout << "number of chunks:" << param.num_chunks << std::endl;
  }

  double project_start = clock();
  if (param.verbose) std::cerr << "start projection" << std::endl;
  std::vector<uint8_t*> sig;
  project_vectors(param.project_dim * param.num_chunks, sig, param);
  if (param.verbose) std::cerr << "end projection" << std::endl;
  double project_end = clock();
  if (param.verbose) std::cout << "projecttime:" << (project_end - project_start)/(double)CLOCKS_PER_SEC << std::endl;

  param.pchunks = new PosRange[param.num_chunks + 1];
  for (int i = 1; i <= (int)param.num_chunks; i++) {
    param.pchunks[i].start = (int)ceil((double)param.seq_len*((double)(i - 1)/(double)param.num_chunks)) + 1;
    param.pchunks[i].end   = (int)ceil((double)param.seq_len*(double)i/(double)param.num_chunks);
  }

  double msm_time = 0.0;

  if (param.verbose) {
    std::cerr << "chunk distance:" << param.chunk_dist << std::endl;
    std::cerr << "the number of blocks:" << param.num_blocks << std::endl;
  }
  param.pos = new PosRange[param.num_blocks + 1];
  for (int i = 1; i <= (int) param.num_chunks; i++) {
    param.chunk_len   = param.pchunks[i].end - param.pchunks[i].start + 1;
    param.start_chunk = param.pchunks[i].start;
    param.cchunk      = i;
    for (int j = 1; j <= (int)param.num_blocks; j++) {
      param.pos[j].start = (int)ceil((double)param.chunk_len*((double)(j - 1)/(double)param.num_blocks)) + param.pchunks[i].start;
      param.pos[j].end   = (int)ceil((double)param.chunk_len*(double)j/(double)param.num_blocks) + param.pchunks[i].start - 1;
    }
    if (param.verbose) std::cerr << "start enumeration chunk no " << i << std::endl;
    double msm_start = clock();
    multi_classification(sig, 1, 0, param.num_seq - 1, param);
    double msm_end   = clock();
    msm_time += (msm_end - msm_start)/(double)CLOCKS_PER_SEC;
  }
  if (param.verbose) std::cout << "msmtime:" << msm_time << std::endl;

  double total_end = clock();
  if (param.verbose) std::cout << "cputime:" << (total_end - total_start)/(double)CLOCKS_PER_SEC << std::endl;

  delete[] param.counter;
  delete[] param.pchunks;
  delete[] param.pos;
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
