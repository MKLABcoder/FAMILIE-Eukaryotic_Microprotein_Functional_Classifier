#!/usr/bin/env python
#_*_coding:utf-8_*_

import os
import json
import joblib
import pandas as pd
from pathlib import Path

from descriptors.AAC import AAC
from descriptors.DPC import DPC
from descriptors.CTDC import CTDC
from descriptors.CTDT import CTDT
from descriptors.CTDD import CTDD
from descriptors.PAAC import PAAC
from descriptors.QSOrder import QSOrder
from descriptors.Moran import Moran


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "models"

CLASS_ORDER = [
    "Immune",
    "Regulation",
    "Metabolism",
    "Secretory",
    "Signalling",
    "Structural",
    "Toxin",
    "Transport"
]

MIN_LEN = 6
MAX_LEN = 100
#load_models()
# =========================================================
# LOAD MODELS
# =========================================================
def load_models():
    print("Loading ML models...")
    models = {}
    thresholds = {}
    features = {}

    for cls in CLASS_ORDER:

        cls_dir = MODEL_ROOT / cls

        if not cls_dir.exists():
            continue

        models[cls] = joblib.load(cls_dir / "model.pkl")

        with open(cls_dir / "thresholds.json") as f:
            thresholds[cls] = json.load(f)

        with open(cls_dir / "features.txt") as f:
            features[cls] = [x.strip() for x in f]

    return models, thresholds, features


models = None
thresholds = None
feature_lists = None

#compute_all_features()
# =========================================================
# FEATURE COMPUTATION
# =========================================================
def compute_all_features(fastas):
    dfs = []

    for desc in [AAC, DPC, CTDC, CTDT, CTDD, PAAC, QSOrder, Moran]:
        enc = desc(fastas)

        if isinstance(enc, int):
            raise ValueError(
                f"{desc.__name__} encoding failed and returned int={enc}"
            )

        df = pd.DataFrame(enc[1:], columns=enc[0])
        df.index = df.iloc[:, 0]
        df = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        df.columns = [f"{desc.__name__}__{c}" for c in df.columns]
        dfs.append(df)

    return pd.concat(dfs, axis=1)

# =========================================================
# Split the FASTA records
# =========================================================
from math import ceil

def split_fasta(fastas, cores):
    chunk_size = ceil(len(fastas) / cores)
    return [
        fastas[i:i + chunk_size]
        for i in range(0, len(fastas), chunk_size)
    ]
# =========================================================
# Worker function
# =========================================================
def feature_worker(chunk):
    return compute_all_features(chunk)

# =========================================================
# Parallel feature computation
# =========================================================
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

def compute_all_features_parallel(fastas, cores):

    chunks = split_fasta(fastas, cores)

    with ProcessPoolExecutor(max_workers=cores) as executor:
        dfs = list(executor.map(feature_worker, chunks))

    return pd.concat(dfs)

#confidence_level()
# =========================================================
# PREDICTION HELPERS
# =========================================================
def confidence_level(prob, thr):
    if prob < thr:
        return "-"

    rel_margin = (prob - thr) / thr

    if rel_margin >= 0.5:
        return "High"
    elif rel_margin >= 0.1:
        return "Medium"
    else:
        return "Low"

#predict_class()
def predict_class(X, cls, mode):
    model = models[cls]
    feats = feature_lists[cls]
    thr = thresholds[cls][mode]

    Xc = X.reindex(columns=feats, fill_value=0)

    probs = model.predict_proba(Xc)[:, 1]
    preds = probs >= thr
    confs = [confidence_level(p, thr) for p in probs]

    return probs, preds, confs

def run_prediction(fasta_records, mode="Validation",cores=1):
    
    global models, thresholds, feature_lists

    if models is None:
        models, thresholds, feature_lists = load_models()
    
    
    valid = []

    for sid, seq in fasta_records:
        if MIN_LEN <= len(seq) <= MAX_LEN:
            valid.append((sid, seq))

    if len(valid) == 0:
        raise ValueError("No valid sequences found.")
    
    print("Computing descriptors...")
    if cores == 1:
        feats = compute_all_features(valid)
    else:
        feats = compute_all_features_parallel(valid, cores)
    
    print("Running prediction...")
    
    results = pd.DataFrame(index=feats.index)

    for cls in CLASS_ORDER:
        p, pr, c = predict_class(feats, cls, mode)
        results[(cls, "Pred")] = pr
        results[(cls, "Probability")] = p
        results[(cls, "Confidence")] = c

    results.columns = pd.MultiIndex.from_tuples(results.columns)
    output = results.copy()

    for cls in CLASS_ORDER:
        output[(cls, "Pred")] = output[(cls, "Pred")].map(
            lambda x: "Yes" if x else "No"
        )

    return output
#banner()
# =========================================================
# banner
# =========================================================
def print_banner():
    print("=" * 60)
    print("FAMILIE v1.0")
    print("Functional Assignment of Microproteins Into Labels In Eukaryotes")
    print("=" * 60)
    print()
