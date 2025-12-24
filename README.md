# FAMILIE - Functional Assignment of Microproteins Into Labels In Eukaryotes

A production-ready machine learning–based web tool designed to predict the functional classes of microproteins using sequence-derived features.
The tool accepts protein sequences as input and assigns probabilistic scores across multiple functional categories, enabling both high-sensitivity screening and high-confidence validation use cases.

## Key Features

- Supports single-sequence input and FASTA file uploads
- Uses sequence-derived descriptors computed via iFeature
- Multi-class functional prediction with class-wise probability scores
- Provides ranked class outputs with downloadable CSV results
- Implements learned probability thresholds for screening and validation modes

## Model Overview

- Trained on reviewed UniProt eukaryotic microproteins
- Feature selection performed using permutation importance and bootstrap-based methods
- Final model optimized for strong ROC-AUC and Precision–Recall performance
- Designed to generalize to unannotated or newly discovered microproteins

## Repository Structure

├── app/                # Web application code
├── models/             # Trained ML models, selected features and thresholds
├── descriptors/        # Code to compute features
├── assets/             # contains logo image
├── requirements.txt    # Python dependencies
└── README.md

### Installaton:
    pip install -r requirements.txt

## Usage

- Provide protein sequences via text input or FASTA upload
- Select prediction mode (Screening or Validation)
- View and download ranked predictons and class memberships as a .csv file

## Intended Use
This tool is intended for research and exploratory analysis of microprotein functions and is not designed for clinical or diagnostic use. 