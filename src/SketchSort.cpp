// SPDX-License-Identifier: MIT
/*
 * SketchSort.cpp
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

#include "SketchSort.hpp"

template<class T>
inline uint8_t sign(T val) {
  if (val > 0)
    return 1;
  return 0;
}

bool cmp(const std::pair<int, float> &p1, const std::pair<int, float> &p2) {
  return p1.second < p2.second;
}

void SketchSort::readFeature(const char *fname) {
  std::ifstream ifs(fname);

  if (!ifs) {
    throw std::runtime_error(std::string("cannot open input file: ") + fname);
  }

  dim             = 0;
  float val       = 0.f;
  std::string line;
  unsigned int lineNo = 0;
  while (std::getline(ifs, line)) {
    ++lineNo;
    if (line.find_first_not_of(" \t\r\n\v\f") == std::string::npos) {
      std::ostringstream msg;
      msg << "line " << lineNo << " is empty or contains only whitespace";
      throw std::runtime_error(msg.str());
    }

    fvs.resize(fvs.size() + 1);
    boost::numeric::ublas::vector<float> &fv = fvs[fvs.size() - 1];
    uint32_t counter = 0;
    std::istringstream is(line);
    if (dim != 0) {
      fv.resize(dim);
      while (is >> val) {
	if (counter >= dim) {
	  std::ostringstream msg;
	  msg << "line " << lineNo << " has more than " << dim
	      << " values, expected dimension " << dim;
	  throw std::runtime_error(msg.str());
	}
	fv[counter++]= val;
      }
      if (!is.eof()) {
	std::ostringstream msg;
	msg << "line " << lineNo << " contains a non-numeric token";
	throw std::runtime_error(msg.str());
      }
      if (counter !=  dim) {
	std::ostringstream msg;
	msg << "dimensions of the input vector should be the same. "
	    << "expected " << dim << " got " << counter
	    << " on line: " << line;
	throw std::runtime_error(msg.str());
      }
    }
    else {
      while (is >> val) {
	fv.resize(counter + 1);
	fv[counter] = val;
	counter++;
      }
      if (!is.eof()) {
	std::ostringstream msg;
	msg << "line " << lineNo << " contains a non-numeric token";
	throw std::runtime_error(msg.str());
      }
      dim = counter;
    }
  }

  if (fvs.empty()) {
    throw std::runtime_error("input file contains no data rows");
  }
}

void SketchSort::setFeaturesFromRaw(const float *data, std::size_t n_rows, std::size_t n_cols) {
  fvs.clear();
  fvs.resize(n_rows);
  for (std::size_t i = 0; i < n_rows; i++) {
    boost::numeric::ublas::vector<float> &fv = fvs[i];
    fv.resize(n_cols);
    const float *row = data + i * n_cols;
    for (std::size_t j = 0; j < n_cols; j++) {
      fv[j] = row[j];
    }
  }
  dim = static_cast<unsigned int>(n_cols);
}

void SketchSort::centeringData() {
  size_t d       = fvs[0].size();
  size_t numData = fvs.size();
  float  mean;
  for (size_t i = 0; i < d; i++) {
    mean = 0.f;
    for (size_t j = 0; j < numData; j++) {
      mean += fvs[j][i];
    }
    mean /= (float)numData;
    for (size_t j = 0; j < numData; j++) {
      fvs[j][i] -= mean;
    }
  }
}

/* sparce random projection
int SketchSort::projectVectors(unsigned int projectDim, std::vector<uint8_t*> &sig, params &param) {

  p = new boost::pool<>(sizeof(uint8_t));
  sig.resize(fvs.size());
  param.ids.resize(fvs.size());
  for (size_t i = 0; i < sig.size(); i++) {
    //    sig[i]    = new uint32_t[projectDim + 1];                                      
    sig[i]    = (uint8_t*)p->ordered_malloc(projectDim + 1);
    param.ids[i] = i;
  }

  boost::mt19937 gen(static_cast<unsigned long>(time(0)));
  boost::uniform_real<> dst(0.f, 1.f);
  boost::variate_generator<boost::mt19937&, boost::uniform_real<> > rand(gen, dst);
  //  double tiny = 1.0/1.79e+308;                                           
  std::vector<std::pair<int, float> > randMat;
  float s = sqrt(float(dim));
  //  float s     = dim/log(dim);
  float thr   = 1.f/(2*s);
  float coff  = sqrt(s);
  for (size_t i = 0; i < projectDim; i++) {
    randMat.clear();
    for (size_t j = 0; j < dim; j++) {
      float r   = rand();
      if       (r < thr) {
        randMat.push_back(std::make_pair(j, coff));
      } else if (r < 2*thr) {	
        randMat.push_back(std::make_pair(j, -coff));
      }
    }

    for (size_t j = 0; j < fvs.size(); j++) {
      boost::numeric::ublas::vector<float> &fv  = fvs[j];
      double proc = 0.f;
      for (size_t k = 0; k < randMat.size(); k++) {
        proc += fv[randMat[k].first] * randMat[k].second;
      }
      sig[j][i+1] = sign(proc);
    }
  }
  param.seq_len = projectDim;
  param.num_seq = fvs.size();

  return 1;
}
*/

int SketchSort::projectVectors(unsigned int projectDim, std::vector<uint8_t*> &sig, params &param) {
  std::vector<float> randMat;
  p = new boost::pool<>(sizeof(uint8_t));
  sig.resize(fvs.size());
  param.ids.resize(fvs.size());
  for (size_t i = 0; i < sig.size(); i++) {
    //    sig[i]    = new uint32_t[projectDim + 1];
    sig[i]    = (uint8_t*)p->ordered_malloc(projectDim + 1);
    param.ids[i] = i;
  }

  boost::mt19937 gen(static_cast<unsigned long>(param.seed));
  boost::normal_distribution<> dst(0.f, 1.f);
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > rand(gen, dst);

  //  double tiny = 1.0/1.79e+308;
  randMat.resize(dim + 1);
  for (size_t i = 0; i < projectDim; i++) {
    for (size_t j = 0; j <= dim; j++) {
      randMat[j] = rand();
    }

    for (size_t j = 0; j < fvs.size(); j++) {
      boost::numeric::ublas::vector<float> &fv  = fvs[j];
      double proc = 0.f;
      for (size_t k = 0; k < fv.size(); k++)
        proc += fv[k] * randMat[k];

      sig[j][i+1] = sign(proc);
    }
  }
  param.seq_len = projectDim;
  param.num_seq = fvs.size();

  return 1;
}

inline float SketchSort::checkCos(unsigned int id1, unsigned int id2) {
  ++numCosDist;
  boost::numeric::ublas::vector<float> &fv_1 = fvs[id1];
  boost::numeric::ublas::vector<float> &fv_2 = fvs[id2];
  float sum = boost::numeric::ublas::inner_prod(fv_1, fv_2);

  return (1.f - sum*(norms[id1]*norms[id2]));
}

inline void SketchSort::sort(std::vector<uint8_t*> &sig, int spos, int epos, int l, int r, params &param) {
   if (r - l + 1 > 50) radixsort(sig, spos, epos, l, r, param);
  else                insertionSort(sig, spos, epos, l, r, param);
}

inline void SketchSort::radixsort(std::vector<uint8_t*> &sig, int spos, int epos, int l, int r, params &param) {
  unsigned int *c      = param.counter;
  std::vector<unsigned int> &ids      = param.ids;
  std::vector<uint8_t*> newsig(r - l + 1);
  std::vector<unsigned int> newids(r - l + 1);
  unsigned int tmp;
  int tpos = spos - 1;
  while (++tpos <= epos) {
    for (int i = 0; i < num_char; i++) *(c + i) = 0;
    for (int i = l; i <= r; i++) c[sig[i][tpos]]++;
    for (int i = 1; i < num_char; i++) *(c + i) += *(c + i - 1);
    for (int i = r; i >= l; --i) {
      tmp = --c[sig[i][tpos]] + l;
      newids[tmp - l] = ids[i];
      newsig[tmp - l] = sig[i];
    }
    if (++tpos <= epos) {
      for (int i = 0; i < num_char; i++) *(c + i) = 0;
      for (int i = l; i <= r; i++) c[newsig[i - l][tpos]]++;
      for (int i = 1; i < num_char; i++) *(c + i) += *(c + i - 1);
      for (int i = r; i >= l; --i) {
	tmp = --c[newsig[i - l][tpos]] + l;
	ids[tmp] = newids[i - l];
	sig[tmp] = newsig[i - l];
      }
    }
    else {
      for (int i = l; i <= r; i++) {
	ids[i] = newids[i - l];
	sig[i] = newsig[i - l];
      }
      return;
    }
  }
}

inline void SketchSort::insertionSort(std::vector<uint8_t*> &sig, int spos, int epos, int l, int r, params &param) {
  int i, j;
  uint8_t *pivot, pval;
  unsigned int pid;
  std::vector<unsigned int> &ids = param.ids;
  for (int tpos = spos; tpos <= epos; tpos++) {
    for (i = l + 1; i <= r; i++) {
      pivot = sig[i]; pval = sig[i][tpos]; pid = ids[i];
      for (j = i; j > l && sig[j-1][tpos] > pval; j--) {
	sig[j]       = sig[j-1];
	ids[j]       = ids[j-1];
      }
      sig[j] = pivot;
      ids[j] = pid;
    }
  }
}

inline void SketchSort::classify(std::vector<uint8_t*> &sig, int spos, int epos, int l, int r, int bpos, params &param) {
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

inline bool SketchSort::calc_chunk_hamdist(uint8_t *seq1, uint8_t *seq2, const params &param) {
  ++numHamDist;
  unsigned int d = 0;
  for (size_t i = 1;  i <= param.chunk_len; i++) 
    if (*seq1++ != *seq2++ && ++d > param.chunk_dist) return false;
  return true;
}

inline bool SketchSort::check_chunk_canonical(uint8_t *seq1, uint8_t *seq2, const params &param) {
  unsigned int d = 0;
  int end        = param.pchunks[param.cchunk].start - 1;
  int j          = 1;
  int tend       = param.pchunks[j].end;
  int i          = 0;
  
  while (++i <= end) {
    if ((d += abs(seq1[i] - seq2[i])) > param.chunk_dist) {
      while (++i <= tend) d += abs(seq1[i] - seq2[i]);
			    //	if (seq1[i] != seq2[i]) ++d;
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

inline bool SketchSort::check_canonical(uint8_t *seq1, uint8_t *seq2, const params &param) {
  size_t sb = 1, eb = 1;
  size_t b;
  for (size_t i = 0, size = param.blocks.size(); i < size; i++) {
    eb = param.blocks[i];
    for (b = sb; b < eb; b++) {
      if (std::equal(seq1 + param.pos[b].start, seq1 + param.pos[b].end + 1, seq2 + param.pos[b].start))
	  return false;
    }
    sb = param.blocks[i] + 1;
  }
  return true;
}


inline void SketchSort::report(std::vector<uint8_t*> &sig, int l, int r, params &param) {
  float cosDist;
  for (int i = l; i < r; i++) {
    for (int j = i + 1; j <= r; j++) {
      if (check_canonical(sig[i], sig[j], param) &&
	  calc_chunk_hamdist(sig[i] + param.start_chunk, sig[j] + param.start_chunk, param) &&
	  check_chunk_canonical(sig[i], sig[j], param) &&
	  ((cosDist = checkCos(param.ids[i], param.ids[j])) <= param.cosDist)) {
	if (param.pairs) {
	  sketchsort_api::Pair pr;
	  pr.id1      = static_cast<std::uint32_t>(param.ids[i]);
	  pr.id2      = static_cast<std::uint32_t>(param.ids[j]);
	  pr.cos_dist = cosDist;
	  param.pairs->push_back(pr);
	}
	if (param.os) {
	  (*param.os) << param.ids[i] << " " << param.ids[j] << " " << cosDist << "\n";
	}
      }
    }
  }
}

void SketchSort::multi_classification(std::vector<uint8_t*> &sig, int maxind, int l, int r, params &param) {
  if (param.blocks.size() == param.numblocks - param.chunk_dist) {
    report(sig, l, r, param);
    return;
  }

  for (int bpos = maxind; bpos <= (int)param.numblocks; bpos++) {
    if (param.blocks.size() + (param.numblocks - bpos + 1) < param.numblocks - param.chunk_dist) { // pruning
      //      std::cerr << "return " << std::endl;
      return;
    }
    param.blocks.push_back(bpos);
    sort(sig, param.pos[bpos].start, param.pos[bpos].end, l, r, param);
    classify(sig, param.pos[bpos].start, param.pos[bpos].end, l, r, bpos, param);
    param.blocks.pop_back();
  }
}

double combination(int n, int m) {
  double sum = 1.0;
  for (int i = 0; i < m; i++) {
    sum *= static_cast<double>(n - i) / static_cast<double>(m - i);
  }
  return sum;
}

double SketchSort::calcMissingEdgeRatio(params &param) {
  double sum = 0.f;
  double prob = acos(1.0 - param.cosDist)/M_PI;
  for (unsigned int k = 0; k <= param.chunk_dist; k++) {
    sum += (combination(param.projectDim, k) * pow(prob, k) * pow(1 - prob, param.projectDim - k));
  }
  return pow(1.0 - sum, param.numchunks);
}

void SketchSort::preComputeNorms(bool centered) {
  norms.resize(fvs.size());
  float sum;
  for (size_t i = 0; i < fvs.size(); i++) {
    boost::numeric::ublas::vector<float> &fv = fvs[i];
    sum = 0.f;
    for (size_t j = 0; j < fv.size(); j++) {
      sum += pow(fv[j], 2);
    }
    if (!(std::isfinite(sum) && sum > 0.f)) {
      std::ostringstream msg;
      msg << "row " << i << " has a zero or non-finite norm"
          << (centered ? " (after centering)" : "")
          << "; cannot compute a cosine distance for it";
      throw std::invalid_argument(msg.str());
    }
    norms[i] = 1.f/sqrt(sum);
  }
}

void SketchSort::decideParameters(float _missingratio, params &param) {
  unsigned int hamDist   = 1;
  unsigned int numBlocks = hamDist + 3;
  unsigned int numchunks = 0;

  do {
    if (numchunks > 30) {
      hamDist   += 1;
      numBlocks  = hamDist + 3;
      numchunks  = 0;
    }
    numchunks += 1;
    param.chunk_dist = hamDist;
    param.numblocks  = numBlocks;
    param.numchunks  = numchunks;
  } while (calcMissingEdgeRatio(param) >= _missingratio);
}

void SketchSort::runCore(params &param) {
  numSort    = 0;
  numCosDist = 0;
  numHamDist = 0;

  if (param.autoFlag) {
    if (!(param.missingratio > 0.f && param.missingratio < 1.f)) {
      std::ostringstream msg;
      msg << "missing_ratio must be in (0, 1), got " << param.missingratio;
      throw std::invalid_argument(msg.str());
    }
    if (param.verbose) std::cerr << "deciding parameters such that the missing edge ratio is no more than " << param.missingratio << std::endl;
    decideParameters(param.missingratio, param);
    if (param.verbose) {
      std::cout << "decided parameters:" << std::endl;
      std::cout << "hamming distance threshold: " << param.chunk_dist << std::endl;
      std::cout << "number of blocks: " << param.numblocks << std::endl;
      std::cout << "number of chunks: "  << param.numchunks << std::endl;
      std::cout << std::endl;
    }
  }

  if (param.numchunks < 1) {
    throw std::invalid_argument("num_chunks must be >= 1");
  }
  if (param.numchunks > (std::numeric_limits<unsigned int>::max)() / param.projectDim) {
    std::ostringstream msg;
    msg << "num_chunks (" << param.numchunks << ") is too large: "
        << param.projectDim << " * num_chunks would overflow";
    throw std::invalid_argument(msg.str());
  }
  if (param.numblocks < 1) {
    std::ostringstream msg;
    msg << "num_blocks must be >= 1, got " << param.numblocks;
    throw std::invalid_argument(msg.str());
  }
  if (param.numblocks > param.projectDim) {
    std::ostringstream msg;
    if (param.autoFlag) {
      msg << "cos_dist is too large: auto mode selected num_blocks=" << param.numblocks
          << ", which exceeds the maximum of " << param.projectDim << ". "
          << "Lower cos_dist or specify ham_dist/num_blocks/num_chunks manually.";
    } else {
      msg << "num_blocks must be <= " << param.projectDim << ", got " << param.numblocks;
    }
    throw std::invalid_argument(msg.str());
  }
  if (param.chunk_dist >= param.numblocks) {
    std::ostringstream msg;
    msg << "ham_dist (" << param.chunk_dist << ") must be less than num_blocks ("
        << param.numblocks << ")";
    throw std::invalid_argument(msg.str());
  }
  if (!(param.cosDist >= 0.f && param.cosDist <= 2.f)) {
    std::ostringstream msg;
    msg << "cos_dist must be in [0, 2], got " << param.cosDist;
    throw std::invalid_argument(msg.str());
  }

  if (param.verbose) std::cout << "missing edge ratio:" << calcMissingEdgeRatio(param) << std::endl;

  if (param.centering) {
    if (param.verbose) std::cerr << "start making input-data centered at 0" << std::endl;
    double centeringstart = clock();
    centeringData();
    double centeringend = clock();
    if (param.verbose) {
      std::cerr << "end making input-data centered at 0" << std::endl;
      std::cout << "centering time:" << (centeringend - centeringstart)/(double)CLOCKS_PER_SEC << std::endl;
    }
  }

  double totalstart = clock();
  preComputeNorms(param.centering);

  param.counter = new unsigned int[num_char];

  if (param.verbose) {
    std::cout << "number of data:" << fvs.size() << std::endl;
    std::cout << "data dimension:" << dim << std::endl;
    std::cout << "projected dimension:" << param.projectDim << std::endl;
    std::cout << "length of strings:" << param.projectDim * param.numchunks << std::endl;
    std::cout << "number of chunks:" << param.numchunks << std::endl;
  }

  double projectstart = clock();
  if (param.verbose) std::cerr << "start projection" << std::endl;
  std::vector<uint8_t*> sig;
  projectVectors(param.projectDim * param.numchunks, sig, param);
  if (param.verbose) std::cerr << "end projection" << std::endl;
  double projectend = clock();
  if (param.verbose) std::cout << "projecttime:" << (projectend - projectstart)/(double)CLOCKS_PER_SEC << std::endl;

  param.pchunks = new pstat[param.numchunks + 1];
  for (int i = 1; i <= (int)param.numchunks; i++) {
    param.pchunks[i].start = (int)ceil((double)param.seq_len*((double)(i - 1)/(double)param.numchunks)) + 1;
    param.pchunks[i].end   = (int)ceil((double)param.seq_len*(double)i/(double)param.numchunks);
  }

  double msmtime = 0.0;

  if (param.verbose) {
    std::cerr << "chunk distance:" << param.chunk_dist << std::endl;
    std::cerr << "the number of blocks:" << param.numblocks << std::endl;
  }
  param.pos = new pstat[param.numblocks + 1];
  for (int i = 1; i <= (int) param.numchunks; i++) {
    param.chunk_len   = param.pchunks[i].end - param.pchunks[i].start + 1;
    param.start_chunk = param.pchunks[i].start;
    param.end_chunk   = param.pchunks[i].end;
    param.cchunk      = i;
    for (int j = 1; j <= (int)param.numblocks; j++) {
      param.pos[j].start = (int)ceil((double)param.chunk_len*((double)(j - 1)/(double)param.numblocks)) + param.pchunks[i].start;
      param.pos[j].end   = (int)ceil((double)param.chunk_len*(double)j/(double)param.numblocks) + param.pchunks[i].start - 1;
    }
    if (param.verbose) std::cerr << "start enumeration chunk no " << i << std::endl;
    double msmstart = clock();
    multi_classification(sig, 1, 0, param.num_seq - 1, param);
    double msmend   = clock();
    msmtime += (msmend - msmstart)/(double)CLOCKS_PER_SEC;
  }
  if (param.verbose) std::cout << "msmtime:" << msmtime << std::endl;

  double totalend = clock();
  if (param.verbose) {
    std::cout << "cputime:" << (totalend - totalstart)/(double)CLOCKS_PER_SEC << std::endl;
    std::cout << "numSort:" << combination(param.numblocks, param.chunk_dist) * param.numchunks << std::endl;
    std::cout << "numHamDist:" << numHamDist << std::endl;
    std::cout << "numCosDist:" << numCosDist << std::endl;
  }

  delete p;
  delete[] param.counter;
  delete[] param.pchunks;
  delete[] param.pos;
}

void SketchSort::run(const char *fname, const char *oname,
		  unsigned int _numblocks,
		  unsigned int _dist,
		  float        _cosDist,
		  unsigned int _numchunks,
		  bool         _autoFlag,
		  float        _missingratio,
		  bool         _centering,
		  unsigned int _seed,
		  bool         _verbose)
{
  params param;
  param.numblocks    = _numblocks;
  param.numchunks    = _numchunks;
  param.chunk_dist   = _dist;
  param.cosDist      = _cosDist;
  param.projectDim   = 32;
  param.autoFlag     = _autoFlag;
  param.missingratio = _missingratio;
  param.centering    = _centering;
  param.seed         = _seed;
  param.verbose      = _verbose;
  num_char           = 2;

  if (param.verbose) std::cerr << "start reading" << std::endl;
  double readstart = clock();
  readFeature(fname);
  double readend   = clock();
  if (param.verbose) {
    std::cerr << "end reading" << std::endl;
    std::cout << "readtime:" << (readend - readstart)/(double)CLOCKS_PER_SEC << std::endl;
  }

  std::ofstream ofs(oname);
  if (!ofs) {
    throw std::runtime_error(std::string("cannot open output file: ") + oname);
  }
  param.os = &ofs;

  runCore(param);

  ofs.close();
}

void SketchSort::search(const float *data, std::size_t n_rows, std::size_t n_cols,
			unsigned int _numblocks,
			unsigned int _dist,
			float        _cosDist,
			unsigned int _numchunks,
			bool         _autoFlag,
			float        _missingratio,
			bool         _centering,
			unsigned int _seed,
			bool         _verbose,
			std::vector<sketchsort_api::Pair> &out)
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

  params param;
  param.numblocks    = _numblocks;
  param.numchunks    = _numchunks;
  param.chunk_dist   = _dist;
  param.cosDist      = _cosDist;
  param.projectDim   = 32;
  param.autoFlag     = _autoFlag;
  param.missingratio = _missingratio;
  param.centering    = _centering;
  param.seed         = _seed;
  param.verbose      = _verbose;
  num_char           = 2;

  out.clear();
  param.pairs = &out;

  setFeaturesFromRaw(data, n_rows, n_cols);

  runCore(param);
}
