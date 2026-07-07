// SPDX-License-Identifier: MIT
/*
 * main.cpp
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

#include <iostream>
#include <cstdlib>
#include <cerrno>
#include <climits>

using sketchsort::SketchSort;

/* Globals */
void usage(int exit_code);
void version();
void parse_parameters (int argc, char **argv);

char *input_path, *output_path;
unsigned int ham_dist      = 1;
unsigned int num_blocks    = 4;
unsigned int num_chunks    = 3;
float        cos_dist      = 0.01;
bool         auto_mode     = false;
float        missing_ratio = 0.0001;
bool         centering     = false;
unsigned int seed          = 0;

int main(int argc, char **argv)
{
  version();

  try {
    parse_parameters(argc, argv);

    SketchSort sketchsort;
    sketchsort.run(input_path, output_path, num_blocks, ham_dist, cos_dist,
                   num_chunks, auto_mode, missing_ratio, centering, seed);
  } catch (const std::exception &e) {
    std::cerr << "sketchsort: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}

void version(){
  std::cerr << "SketchSort version 0.3.0" << std::endl
            << "Written by Yasuo Tabei" << std::endl << std::endl;
}

void usage(int exit_code){
  std::cerr << std::endl
       << "Usage: sketchsort [OPTION]... INFILE OUTFILE" << std::endl << std::endl
       << "       where [OPTION]...  is a list of zero or more optional arguments" << std::endl
       << "             INFILE       is the name of an input file" << std::endl
       << "             OUTFILE      is the name of an output file" << std::endl << std::endl
       << "Additional arguments (input and output files may be specified):" << std::endl
       << "       -hamdist [maximum hamming distance]" << std::endl
       << "       (default: " << ham_dist << ")" << std::endl
       << "       -numblocks [the number of blocks]" << std::endl
       << "       (default: " << num_blocks << ")" << std::endl
       << "       -cosdist [maximum cosine distance]" << std::endl
       << "       (default: " << cos_dist << ")" << std::endl
       << "       -numchunks [the number of chunks]" << std::endl
       << "       (default: " << num_chunks << ")" << std::endl
       << "       -auto " << std::endl
       << "       -missingratio " << std::endl
       << "       (default: " << missing_ratio << ")" << std::endl
       << "       -centering" << std::endl
       << "       -seed [random seed]" << std::endl
       << "       (default: " << seed << ")" << std::endl
       << "       -h, --help" << std::endl
       << std::endl;
  exit(exit_code);
}

namespace {

float parse_float_arg(const char *flag, const char *s) {
  errno = 0;
  char *endptr = nullptr;
  float v = strtof(s, &endptr);
  if (endptr == s || *endptr != '\0' || errno == ERANGE) {
    std::cerr << "sketchsort: invalid number \"" << s << "\" for " << flag << std::endl;
    exit(1);
  }
  return v;
}

unsigned int parse_uint_arg(const char *flag, const char *s) {
  errno = 0;
  char *endptr = nullptr;
  unsigned long v = strtoul(s, &endptr, 10);
  if (endptr == s || *endptr != '\0' || errno == ERANGE || v > (std::numeric_limits<unsigned int>::max)()) {
    std::cerr << "sketchsort: invalid unsigned integer value \"" << s << "\" for " << flag << std::endl;
    exit(1);
  }
  return static_cast<unsigned int>(v);
}

} // namespace

void parse_parameters (int argc, char **argv){
  if (argc == 1) usage(1);
  int argno;
  for (argno = 1; argno < argc; argno++){
    if (argv[argno][0] == '-'){
      if      (!strcmp (argv[argno], "-version")){
	exit(0);
      }
      else if (!strcmp (argv[argno], "-h") || !strcmp (argv[argno], "--help")) {
	usage(0);
      }
      else if (!strcmp (argv[argno], "-auto")) {
	auto_mode = true;
      }
      else if (!strcmp (argv[argno], "-centering")) {
	centering = true;
      }
      else if (!strcmp (argv[argno], "-numblocks")) {
	if (argno == argc - 1) {
	  std::cerr << "sketchsort: must specify a value after -numblocks" << std::endl;
	  exit(1);
	}
	num_blocks = parse_uint_arg("-numblocks", argv[++argno]);
      }
      else if (!strcmp (argv[argno], "-hamdist")) {
	if (argno == argc - 1) {
	  std::cerr << "sketchsort: must specify a value after -hamdist" << std::endl;
	  exit(1);
	}
	ham_dist = parse_uint_arg("-hamdist", argv[++argno]);
      }
      else if (!strcmp (argv[argno], "-cosdist")) {
	if (argno == argc - 1) {
	  std::cerr << "sketchsort: must specify a value after -cosdist" << std::endl;
	  exit(1);
	}
	cos_dist = parse_float_arg("-cosdist", argv[++argno]);
      }
      else if (!strcmp (argv[argno], "-numchunks")) {
	if (argno == argc - 1) {
	  std::cerr << "sketchsort: must specify a value after -numchunks" << std::endl;
	  exit(1);
	}
	num_chunks = parse_uint_arg("-numchunks", argv[++argno]);
      }
      else if (!strcmp (argv[argno], "-missingratio")) {
	if (argno == argc - 1) {
	  std::cerr << "sketchsort: must specify a value after -missingratio" << std::endl;
	  exit(1);
	}
	missing_ratio = parse_float_arg("-missingratio", argv[++argno]);
      }
      else if (!strcmp (argv[argno], "-seed")) {
	if (argno == argc - 1) {
	  std::cerr << "sketchsort: must specify a value after -seed" << std::endl;
	  exit(1);
	}
	seed = parse_uint_arg("-seed", argv[++argno]);
      }
      else {
	std::cerr << "sketchsort: unknown option " << argv[argno] << std::endl;
	usage(1);
      }
    } else {
      break;
    }
  }

  if (argc - argno != 2) {
    if (argc - argno < 2) {
      std::cerr << "sketchsort: missing "
                << (argc - argno == 0 ? "INFILE and OUTFILE" : "OUTFILE") << std::endl;
    } else {
      std::cerr << "sketchsort: too many arguments after OUTFILE "
                   "(options must precede INFILE)" << std::endl;
    }
    usage(1);
  }

  input_path = argv[argno];
  output_path = argv[argno + 1];
}
