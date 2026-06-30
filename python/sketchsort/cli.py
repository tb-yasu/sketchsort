# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SketchSort contributors
"""Command-line entry point matching the legacy C++ sketchsort CLI.

The argument names and the output file format are kept byte-identical with the
C++ binary (src/Main.cpp + src/SketchSort.cpp) by delegating to the C++ side
via run_from_file().
"""

import argparse
import sys

from ._core import run_from_file


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sketchsort",
        description="Fast all-pairs cosine-similarity search via random projection sketches.",
        add_help=True,
    )
    p.add_argument("-hamdist", type=int, default=1,
                   help="maximum Hamming distance (default: 1)")
    p.add_argument("-numblocks", type=int, default=4,
                   help="number of blocks (default: 4)")
    p.add_argument("-cosdist", type=float, default=0.01,
                   help="maximum cosine distance (default: 0.01)")
    p.add_argument("-numchunks", type=int, default=3,
                   help="number of chunks (default: 3)")
    p.add_argument("-auto", dest="auto_mode", action="store_true",
                   help="derive ham_dist / num_blocks / num_chunks from -missingratio")
    p.add_argument("-missingratio", type=float, default=0.0001,
                   help="target missing edge ratio used with -auto (default: 0.0001)")
    p.add_argument("-centering", dest="centering", action="store_true",
                   help="subtract per-dimension mean before sketching")
    p.add_argument("-seed", type=int, default=0,
                   help="RNG seed (default: 0)")
    p.add_argument("-quiet", dest="quiet", action="store_true",
                   help="suppress algorithm progress messages on stdout/stderr")
    p.add_argument("infile", help="input file: one whitespace-separated float vector per line")
    p.add_argument("outfile", help="output file: 'id1 id2 cos_dist' per line")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        run_from_file(
            input_path=args.infile,
            output_path=args.outfile,
            cos_dist=args.cosdist,
            ham_dist=args.hamdist,
            num_blocks=args.numblocks,
            num_chunks=args.numchunks,
            auto_mode=args.auto_mode,
            missing_ratio=args.missingratio,
            centering=args.centering,
            seed=args.seed,
            verbose=not args.quiet,
        )
    except (RuntimeError, OSError, ValueError) as e:
        print(f"sketchsort: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
