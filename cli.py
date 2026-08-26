#!/usr/bin/env python
#_*_coding:utf-8_*_


import argparse
from core.io import read_fasta
from core.predictor import run_prediction, print_banner
def main():
    print_banner()
    import time
    start = time.perf_counter()
    parser = argparse.ArgumentParser(prog="familie",
                                     description= " Functional Assignment of Microproteins Into Labels In Eukaryotes")

    parser.add_argument("--input", required=True, help="Microproteins fasta file")
    parser.add_argument(
        "--mode",
        choices=["Screening","Validation"],
        default="Validation"
    )

    parser.add_argument(
        "--output",
        default="results.csv",
        help="a CSV file"
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Number of CPU cores"
    )
    parser.add_argument(
    "--version",
    action="version",
    version="FAMILIE 1.0"
    )
    args = parser.parse_args()

    fastas = read_fasta(args.input)
    
    print(f"Input FASTA      : {args.input}")
    print(f"Output File      : {args.output}")
    print(f"Prediction Mode  : {args.mode}")
    print(f"CPU Cores        : {args.cores}")
    print(f"Sequences Loaded : {len(fastas)}")
    print()

    results = run_prediction(
        fastas,
        mode=args.mode,
        cores=args.cores
    )
    print("Writing results...")
    results.to_csv(args.output)
    print("\nPrediction Summary")
    print("------------------")
    print(f"Sequences processed : {len(fastas)}")
    print(f"Output saved to     : {args.output}")
    print("✓ Results written successfully.")
    elapsed = time.perf_counter() - start
    print(f"\nFinished successfully.")
    print(f"Total runtime : {elapsed:.2f} seconds")
if __name__== "__main__":
    main()

