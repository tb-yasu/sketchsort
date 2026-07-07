// SPDX-License-Identifier: MIT
/*
 * multi_sort_engine.hpp
 * Copyright (c) 2011 Yasuo Tabei
 *
 * The multiple-sort all-pairs enumeration engine shared by the cosine,
 * min-max, and Jaccard cores (Tabei, Uno, Sugiyama, Tsuda: "Single versus
 * Multiple Sorting in All Pairs Similarity Search", ACML 2010).
 *
 * Every metric sketches its input into byte strings of length
 * project_dim * num_chunks (one hash value or sign bit per byte, 1-indexed
 * with slot 0 unused). This engine is metric-agnostic: it enumerates, for
 * every chunk, all block combinations of size num_blocks - ham_dist,
 * radix/insertion-sorts the sketches on the selected block bytes, groups
 * equal keys, applies the canonicality checks that guarantee each pair is
 * examined exactly once, and hands surviving candidate pairs to a
 * metric-specific Emitter:
 *
 *   struct Emitter {
 *     // Cheap O(1) prefilter (e.g. a set-size bound); return false to skip
 *     // the pair before any sketch comparison. Return true if unused.
 *     bool prefilter(unsigned int id1, unsigned int id2);
 *     // Exact-distance verification and output. Called once per candidate
 *     // pair that passed the hamming and canonicality filters.
 *     void candidate(unsigned int id1, unsigned int id2);
 *   };
 *
 * The Emitter is a template parameter so both hooks inline; the enumeration
 * hot loops (counting sort, block equality, hamming scan) contain no
 * metric-specific code at all.
 */

#ifndef SKETCHSORT_MULTI_SORT_ENGINE_HPP
#define SKETCHSORT_MULTI_SORT_ENGINE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace sketchsort {
namespace engine {

struct PosRange {
  int start;
  int end;
};

struct Config {
  unsigned int num_blocks;
  unsigned int num_chunks;
  unsigned int ham_dist;       // allowed mismatches per chunk
  unsigned int alphabet_size;  // 2 for sign sketches, 256 for byte hashes
  unsigned int sketch_len;     // project_dim * num_chunks
  bool         verbose;
};

// ---------------------------------------------------------------------------
// Shared parameter mathematics
// ---------------------------------------------------------------------------

inline double combination(int n, int m) {
  double sum = 1.0;
  for (int i = 0; i < m; i++) {
    sum *= static_cast<double>(n - i) / static_cast<double>(m - i);
  }
  return sum;
}

// Probability that a pair at the metric's distance threshold survives no
// chunk, given the per-position sketch mismatch probability at that
// threshold (metric-specific: acos(1-d)/pi for sign sketches, (255/256)*d
// for 8-bit hashes).
inline double missing_edge_ratio(double mismatch_prob,
                                 unsigned int project_dim,
                                 unsigned int ham_dist,
                                 unsigned int num_chunks) {
  double sum = 0.0;
  for (unsigned int k = 0; k <= ham_dist; k++) {
    sum += combination(project_dim, k) * std::pow(mismatch_prob, k)
           * std::pow(1.0 - mismatch_prob, project_dim - k);
  }
  return std::pow(1.0 - sum, num_chunks);
}

struct AutoParams {
  unsigned int ham_dist;
  unsigned int num_blocks;
  unsigned int num_chunks;
};

// Smallest enumeration parameters whose missing-edge ratio is below
// missing_ratio: grow num_chunks up to 30, then bump ham_dist and retry.
inline AutoParams decide_parameters(double mismatch_prob,
                                    float missing_ratio,
                                    unsigned int project_dim) {
  unsigned int ham_dist   = 1;
  unsigned int num_blocks = ham_dist + 3;
  unsigned int num_chunks = 0;

  do {
    if (num_chunks > 30) {
      ham_dist  += 1;
      num_blocks = ham_dist + 3;
      num_chunks = 0;
    }
    num_chunks += 1;
  } while (missing_edge_ratio(mismatch_prob, project_dim, ham_dist, num_chunks)
           >= missing_ratio);

  return AutoParams{ham_dist, num_blocks, num_chunks};
}

// Common validation of the enumeration parameters plus the metric's distance
// threshold range. Throws std::invalid_argument (Python: ValueError).
inline void validate_params(unsigned int num_blocks,
                            unsigned int num_chunks,
                            unsigned int ham_dist,
                            unsigned int project_dim,
                            bool auto_mode,
                            float dist, float dist_lo, float dist_hi,
                            const char *dist_name) {
  if (num_chunks < 1) {
    throw std::invalid_argument("num_chunks must be >= 1");
  }
  // Extra parens around `max` defeat the Windows <windows.h> `max` macro.
  if (num_chunks > (std::numeric_limits<unsigned int>::max)() / project_dim) {
    std::ostringstream msg;
    msg << "num_chunks (" << num_chunks << ") is too large: "
        << project_dim << " * num_chunks would overflow";
    throw std::invalid_argument(msg.str());
  }
  if (num_blocks < 1) {
    std::ostringstream msg;
    msg << "num_blocks must be >= 1, got " << num_blocks;
    throw std::invalid_argument(msg.str());
  }
  if (num_blocks > project_dim) {
    std::ostringstream msg;
    if (auto_mode) {
      msg << dist_name << " is too large: auto mode selected num_blocks=" << num_blocks
          << ", which exceeds the maximum of " << project_dim << ". "
          << "Lower " << dist_name << " or specify ham_dist/num_blocks/num_chunks manually.";
    } else {
      msg << "num_blocks must be <= " << project_dim << ", got " << num_blocks;
    }
    throw std::invalid_argument(msg.str());
  }
  if (ham_dist >= num_blocks) {
    std::ostringstream msg;
    msg << "ham_dist (" << ham_dist << ") must be less than num_blocks ("
        << num_blocks << ")";
    throw std::invalid_argument(msg.str());
  }
  if (!(dist >= dist_lo && dist <= dist_hi)) {
    std::ostringstream msg;
    msg << dist_name << " must be in [" << dist_lo << ", " << dist_hi
        << "], got " << dist;
    throw std::invalid_argument(msg.str());
  }
}

// ---------------------------------------------------------------------------
// The enumeration engine
// ---------------------------------------------------------------------------

template <class Emitter>
class MultiSortEngine {
 public:
  struct Stats {
    std::uint64_t ham_checks = 0;  // chunk hamming-distance evaluations
    std::uint64_t candidates = 0;  // pairs handed to Emitter::candidate
  };

  MultiSortEngine(const Config &cfg,
                  std::vector<uint8_t*> &sketches,
                  Emitter &emitter)
      : cfg_(cfg), sketches_(sketches), emitter_(emitter) {}

  // Full enumeration over all chunks. sketches_[i][1..sketch_len] must hold
  // row i's sketch; the engine permutes `sketches_` (and its parallel id
  // array) freely while sorting.
  void run() {
    const int n = static_cast<int>(sketches_.size());

    ids_.resize(n);
    for (int i = 0; i < n; ++i) ids_[i] = static_cast<unsigned int>(i);
    counter_.assign(cfg_.alphabet_size, 0u);
    scratch_sketches_.resize(n);
    scratch_ids_.resize(n);
    blocks_.clear();

    pchunks_.assign(cfg_.num_chunks + 1, PosRange());
    for (int i = 1; i <= static_cast<int>(cfg_.num_chunks); i++) {
      pchunks_[i].start = (int)std::ceil((double)cfg_.sketch_len * ((double)(i - 1) / (double)cfg_.num_chunks)) + 1;
      pchunks_[i].end   = (int)std::ceil((double)cfg_.sketch_len * (double)i / (double)cfg_.num_chunks);
    }

    pos_.assign(cfg_.num_blocks + 1, PosRange());
    for (int i = 1; i <= static_cast<int>(cfg_.num_chunks); i++) {
      chunk_len_     = static_cast<unsigned int>(pchunks_[i].end - pchunks_[i].start + 1);
      chunk_start_   = static_cast<unsigned int>(pchunks_[i].start);
      current_chunk_ = static_cast<unsigned int>(i);
      for (int j = 1; j <= static_cast<int>(cfg_.num_blocks); j++) {
        pos_[j].start = (int)std::ceil((double)chunk_len_ * ((double)(j - 1) / (double)cfg_.num_blocks)) + pchunks_[i].start;
        pos_[j].end   = (int)std::ceil((double)chunk_len_ * (double)j / (double)cfg_.num_blocks) + pchunks_[i].start - 1;
      }
      if (cfg_.verbose) std::cerr << "start enumeration chunk no " << i << std::endl;
      multi_classification(1, 0, n - 1);
    }
  }

  const Stats &stats() const { return stats_; }

 private:
  // Recursively pick the next selected block; once num_blocks - ham_dist
  // blocks agree, every pair inside the group is a candidate.
  void multi_classification(int start_block, int left, int right) {
    if (blocks_.size() == cfg_.num_blocks - cfg_.ham_dist) {
      report(left, right);
      return;
    }

    for (int block_idx = start_block; block_idx <= static_cast<int>(cfg_.num_blocks); block_idx++) {
      if (blocks_.size() + (cfg_.num_blocks - block_idx + 1) < cfg_.num_blocks - cfg_.ham_dist) { // pruning
        return;
      }
      blocks_.push_back(block_idx);
      sort_range(pos_[block_idx].start, pos_[block_idx].end, left, right);
      classify(pos_[block_idx].start, pos_[block_idx].end, left, right, block_idx);
      blocks_.pop_back();
    }
  }

  // Split the sorted range into runs of equal block keys and recurse on
  // every run of size >= 2.
  void classify(int start_pos, int end_pos, int left, int right, int block_idx) {
    int run_start = left;
    for (int iter = left + 1; iter <= right; iter++) {
      if (!std::equal(sketches_[run_start] + start_pos, sketches_[run_start] + end_pos + 1,
                      sketches_[iter] + start_pos)) {
        int run_end = iter - 1;
        if (run_end - run_start >= 1)
          multi_classification(block_idx + 1, run_start, run_end);
        run_start = iter;
      }
    }
    if (right - run_start >= 1)
      multi_classification(block_idx + 1, run_start, right);
  }

  void report(int left, int right) {
    for (int i = left; i < right; i++) {
      for (int j = i + 1; j <= right; j++) {
        if (!emitter_.prefilter(ids_[i], ids_[j]))
          continue;
        if (check_canonical(sketches_[i], sketches_[j]) &&
            check_chunk_ham_dist(sketches_[i] + chunk_start_, sketches_[j] + chunk_start_) &&
            check_chunk_canonical(sketches_[i], sketches_[j])) {
          ++stats_.candidates;
          emitter_.candidate(ids_[i], ids_[j]);
        }
      }
    }
  }

  // Small ranges are cheaper to insertion-sort than to radix-sort (which
  // clears an alphabet-sized histogram on every pass). Both sorts are
  // stable position-by-position passes, so they produce identical
  // permutations — the threshold affects speed only.
  void sort_range(int start_pos, int end_pos, int left, int right) {
    if (right - left + 1 > 50) radix_sort(start_pos, end_pos, left, right);
    else                       insertion_sort(start_pos, end_pos, left, right);
  }

  void radix_sort(int start_pos, int end_pos, int left, int right) {
    unsigned int *counts = counter_.data();
    std::vector<unsigned int> &row_ids = ids_;
    std::vector<uint8_t*> &new_sketches = scratch_sketches_;
    std::vector<unsigned int> &new_ids  = scratch_ids_;
    const unsigned int alphabet_size = cfg_.alphabet_size;
    unsigned int dst;
    int byte_pos = start_pos - 1;
    while (++byte_pos <= end_pos) {
      for (unsigned int i = 0; i < alphabet_size; i++) *(counts + i) = 0;
      for (int i = left; i <= right; i++) counts[sketches_[i][byte_pos]]++;
      for (unsigned int i = 1; i < alphabet_size; i++) *(counts + i) += *(counts + i - 1);
      for (int i = right; i >= left; --i) {
        dst = --counts[sketches_[i][byte_pos]] + left;
        new_ids[dst - left] = row_ids[i];
        new_sketches[dst - left] = sketches_[i];
      }
      if (++byte_pos <= end_pos) {
        for (unsigned int i = 0; i < alphabet_size; i++) *(counts + i) = 0;
        for (int i = left; i <= right; i++) counts[new_sketches[i - left][byte_pos]]++;
        for (unsigned int i = 1; i < alphabet_size; i++) *(counts + i) += *(counts + i - 1);
        for (int i = right; i >= left; --i) {
          dst = --counts[new_sketches[i - left][byte_pos]] + left;
          row_ids[dst] = new_ids[i - left];
          sketches_[dst] = new_sketches[i - left];
        }
      }
      else {
        for (int i = left; i <= right; i++) {
          row_ids[i] = new_ids[i - left];
          sketches_[i] = new_sketches[i - left];
        }
        return;
      }
    }
  }

  void insertion_sort(int start_pos, int end_pos, int left, int right) {
    int i, j;
    uint8_t *pivot, pivot_val;
    unsigned int pivot_id;
    std::vector<unsigned int> &row_ids = ids_;
    for (int byte_pos = start_pos; byte_pos <= end_pos; byte_pos++) {
      for (i = left + 1; i <= right; i++) {
        pivot = sketches_[i]; pivot_val = sketches_[i][byte_pos]; pivot_id = row_ids[i];
        for (j = i; j > left && sketches_[j-1][byte_pos] > pivot_val; j--) {
          sketches_[j] = sketches_[j-1];
          row_ids[j]   = row_ids[j-1];
        }
        sketches_[j] = pivot;
        row_ids[j]   = pivot_id;
      }
    }
  }

  bool check_chunk_ham_dist(const uint8_t *sketch1, const uint8_t *sketch2) {
    ++stats_.ham_checks;
    unsigned int dist = 0;
    for (std::size_t i = 1; i <= chunk_len_; i++)
      if (*sketch1++ != *sketch2++ && ++dist > cfg_.ham_dist) return false;
    return true;
  }

  // Reject the pair if any unselected block below a selected one already
  // agrees — that combination would have reported the pair earlier.
  bool check_canonical(const uint8_t *sketch1, const uint8_t *sketch2) const {
    std::size_t start_block = 1, end_block = 1;
    std::size_t b;
    for (std::size_t i = 0, size = blocks_.size(); i < size; i++) {
      end_block = blocks_[i];
      for (b = start_block; b < end_block; b++) {
        if (std::equal(sketch1 + pos_[b].start, sketch1 + pos_[b].end + 1,
                       sketch2 + pos_[b].start))
          return false;
      }
      start_block = blocks_[i] + 1;
    }
    return true;
  }

  // Reject the pair if an earlier chunk already stays within ham_dist —
  // that chunk's enumeration already reported it.
  bool check_chunk_canonical(const uint8_t *sketch1, const uint8_t *sketch2) const {
    unsigned int dist = 0;
    int scan_end  = pchunks_[current_chunk_].start - 1;
    int j         = 1;
    int chunk_end = pchunks_[j].end;
    int i         = 0;

    while (++i <= scan_end) {
      if (sketch1[i] != sketch2[i] && ++dist > cfg_.ham_dist) {
        // This earlier chunk exceeds the threshold, so it cannot cause a
        // duplicate report; skip to the start of the next chunk.
        dist = 0;
        chunk_end = pchunks_[++j].end;
        i = pchunks_[j].start - 1;
        continue;
      }
      if (chunk_end == i)
        return false;
    }
    return true;
  }

  Config cfg_;
  std::vector<uint8_t*> &sketches_;
  Emitter &emitter_;
  Stats stats_;

  std::vector<unsigned int> ids_;        // row id parallel to sketches_
  std::vector<int>          blocks_;     // currently selected block indices
  std::vector<PosRange>     pchunks_;    // 1-indexed chunk byte ranges
  std::vector<PosRange>     pos_;        // 1-indexed block byte ranges (current chunk)
  std::vector<unsigned int> counter_;    // radix histogram
  std::vector<uint8_t*>     scratch_sketches_;  // radix double buffers,
  std::vector<unsigned int> scratch_ids_;       // allocated once and reused
  unsigned int chunk_len_     = 0;
  unsigned int chunk_start_   = 0;
  unsigned int current_chunk_ = 0;
};

}  // namespace engine
}  // namespace sketchsort

#endif  // SKETCHSORT_MULTI_SORT_ENGINE_HPP
