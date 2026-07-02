# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SketchSort contributors
"""Command-line entry point.

Argument names mirror the legacy C++ sketchsort CLI for back-compat. The
new behaviour: if `-hamdist` / `-numblocks` / `-numchunks` are NOT given,
the run defaults to **auto mode** (the library picks them from
`-missingratio`). Pass any of those three to switch to manual mode. The
`-auto` flag is kept as an explicit override (forces auto mode even when
manual params are given).
"""

import argparse
import sys

from . import run_from_file


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sketchsort",
        description="Fast all-pairs cosine-similarity search via random projection sketches.",
        add_help=True,
    )
    p.add_argument("-cosdist", type=float, default=0.01,
                   help="maximum cosine distance (default: 0.01)")
    p.add_argument("-missingratio", type=float, default=0.0001,
                   help="target missing-edge ratio used in auto mode (default: 0.0001)")
    p.add_argument("-hamdist", type=int, default=None,
                   help="manual override: Hamming distance per chunk")
    p.add_argument("-numblocks", type=int, default=None,
                   help="manual override: number of blocks per chunk")
    p.add_argument("-numchunks", type=int, default=None,
                   help="manual override: number of independent sketch chunks")
    p.add_argument("-auto", dest="force_auto", action="store_true",
                   help="force auto mode even if -hamdist/-numblocks/-numchunks are given")
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
        # When force_auto is set, blank the manual knobs so the wrapper
        # selects auto mode. Otherwise pass whatever the user gave through
        # to the wrapper, which auto-detects.
        if args.force_auto:
            ham_dist = num_blocks = num_chunks = None
        else:
            ham_dist   = args.hamdist
            num_blocks = args.numblocks
            num_chunks = args.numchunks

        run_from_file(
            input_path=args.infile,
            output_path=args.outfile,
            cos_dist=args.cosdist,
            ham_dist=ham_dist,
            num_blocks=num_blocks,
            num_chunks=num_chunks,
            missing_ratio=args.missingratio,
            centering=args.centering,
            seed=args.seed,
            verbose=not args.quiet,
        )
    except (RuntimeError, OSError, ValueError, TypeError, OverflowError) as e:
        print(f"sketchsort: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
