import streamlit as st
import pandas as pd
import joblib
import json
import os
import io
from Bio import SeqIO
import plotly.express as px
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))


from descriptors.AAC import AAC
from descriptors.DPC import DPC
from descriptors.CTDC import CTDC
from descriptors.CTDT import CTDT
from descriptors.CTDD import CTDD
from descriptors.PAAC import PAAC
from descriptors.QSOrder import QSOrder
from descriptors.Moran import Moran


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="FAMILIE – Microprotein Functional Classifier",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================
for key in [
    "results",
    "display_results",
    "membership_df",
    "last_mode",
    "last_input_hash",
    "skipped_df"
]:
    if key not in st.session_state:
        st.session_state[key] = None


# =========================================================
# CONSTANTS
# =========================================================
MODEL_ROOT = "models"
MAX_TEXT_SEQUENCES = 200
MIN_LEN = 6
MAX_LEN = 100

CLASS_ORDER = [
    "Immune", "Regulation", "Metabolism", "Secretory",
    "Signalling", "Structural", "Toxin", "Transport"
]

CLASS_COLORS = {
    "Immune": "#3B5B92",        
    "Regulation": "#4E9F9F",    
    "Metabolism": "#7A8F3A",    
    "Secretory": "#C17C54",     
    "Signalling": "#8C5E7A",    
    "Structural": "#5F9E6E",    
    "Toxin": "#B07AA1",         
    "Transport": "#D4A65F",      
    "No_Function": "#6A6A6A" 
}

# =========================================================
# HEADER
# =========================================================
col1, col2 = st.columns([1, 6])
with col1:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", width=120)

with col2:
    st.markdown(
        """
        <h1 style="margin-bottom:0;">FAMILIE – Functional Assignment of Microproteins Into Labels In Eukaryotes</h1>
        <p style="color:gray; font-size:16px;">
        Machine-learning based functional classification of microproteins
        </p>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    f"""
### About
This tool predicts **functional classes of microproteins** using **sequence-derived descriptors** and **pre-trained machine-learning models**.

**Supported Classes:**
- Immune
- Metabolism
- Regulation
- Secretory
- Signalling
- Structural
- Toxin
- Transport

**Prediction modes**
- **Screening** mode for high recall (exploratory)
- **Validation** mode for high confidence (stringent)

**NOTE:**
**This tool is designed for microproteins with sequence length between {MIN_LEN} and {MAX_LEN} amino acids.** Ensure that the uploaded sequences are within these limits.
"""
)


# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():
    models, thresholds, features = {}, {}, {}

    for cls in CLASS_ORDER:
        cls_dir = os.path.join(MODEL_ROOT, cls)
        if not os.path.isdir(cls_dir):
            continue

        models[cls] = joblib.load(os.path.join(cls_dir, "model.pkl"))
        thresholds[cls] = json.load(open(os.path.join(cls_dir, "thresholds.json")))
        features[cls] = [
            f.strip() for f in open(os.path.join(cls_dir, "features.txt"))
        ]

    return models, thresholds, features


models, thresholds, feature_lists = load_models()


# =========================================================
# INPUT
# =========================================================
st.sidebar.header("Input")

input_type = st.sidebar.radio(
    "Sequence Input Type",
    ["FASTA File", "Paste FASTA"]
)

raw_fastas = []

if input_type == "FASTA File":
    f = st.sidebar.file_uploader("Upload FASTA", type=["fa", "fasta"])
    if f:
        records = list(SeqIO.parse(io.StringIO(f.getvalue().decode()), "fasta"))
        raw_fastas = [(r.id, str(r.seq)) for r in records]

else:
    txt = st.sidebar.text_area("Paste FASTA", height=220)
    if txt:
        records = list(SeqIO.parse(io.StringIO(txt), "fasta"))

        if len(records) > MAX_TEXT_SEQUENCES:
            st.sidebar.warning(
                f"Only the first {MAX_TEXT_SEQUENCES} sequences will be used."
            )

        raw_fastas = [(r.id, str(r.seq)) for r in records[:MAX_TEXT_SEQUENCES]]


# =========================================================
# FILTER BY LENGTH
# =========================================================
fastas = []
skipped = []   # list of dicts with id + sequence

for sid, seq in raw_fastas:
    L = len(seq)
    if MIN_LEN <= L <= MAX_LEN:
        fastas.append((sid, seq))
    else:
        skipped.append({
            "Sequence_ID": sid,
            "Sequence": seq,
        })



# =========================================================
# FEATURE COMPUTATION
# =========================================================
@st.cache_data
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


def predict_class(X, cls, mode):
    model = models[cls]
    feats = feature_lists[cls]
    thr = thresholds[cls][mode]

    Xc = X.reindex(columns=feats, fill_value=0)

    probs = model.predict_proba(Xc)[:, 1]
    preds = probs >= thr
    confs = [confidence_level(p, thr) for p in probs]

    return probs, preds, confs


# =========================================================
# MAIN LOGIC
# =========================================================
mode = st.radio(
    "Prediction Mode",
    ["Screening", "Validation"],
    horizontal=True
)

run_clicked = st.button("Run Prediction")

input_hash = hash(tuple(fastas)) if fastas else None

should_run = (
    fastas
    and (
        run_clicked
        or st.session_state.last_mode != mode
        or st.session_state.last_input_hash != input_hash
    )
)

if should_run:

    feats = compute_all_features(fastas)
    results = pd.DataFrame(index=feats.index)

    for cls in CLASS_ORDER:
        p, pr, c = predict_class(feats, cls, mode)
        results[(cls, "Pred")] = pr
        results[(cls, "Probability")] = p
        results[(cls, "Confidence")] = c

    results.columns = pd.MultiIndex.from_tuples(results.columns)

    display = results.copy()
    for cls in CLASS_ORDER:
        display[(cls, "Pred")] = display[(cls, "Pred")].map(
            lambda x: "Yes" if x else "No"
        )

    membership = []
    for sid in results.index:
        assigned = [
            cls for cls in CLASS_ORDER
            if results.loc[sid, (cls, "Pred")]
        ]
        membership.append({
            "Sequence_ID": sid,
            "Predicted_Classes": ", ".join(assigned) if assigned else "None"
        })

    st.session_state.results = results
    st.session_state.display_results = display
    st.session_state.membership_df = pd.DataFrame(membership)
    st.session_state.skipped_df = pd.DataFrame(skipped)
    st.session_state.last_mode = mode
    st.session_state.last_input_hash = input_hash


# =========================================================
# DISPLAY
# =========================================================
if st.session_state.results is not None:

    st.info(f"Results shown for **{mode} mode**")

    if st.session_state.skipped_df is not None and not st.session_state.skipped_df.empty:

        skipped_df = st.session_state.skipped_df.copy()
        skipped_df.insert(0, "S.No.", range(1, len(skipped_df) + 1))

        st.warning(
            f"{len(skipped_df)} sequence(s) were skipped because their length was "
            f"not between {MIN_LEN} and {MAX_LEN} amino acids."
        )

        with st.expander("View skipped sequences"):
            st.dataframe(
                skipped_df,
                use_container_width=True,
                hide_index=True
            )

        st.download_button(
            label="Download skipped sequences (CSV)",
            data=skipped_df.to_csv(index=False),
            file_name="skipped_sequences.csv",
            mime="text/csv"
        )

    results = st.session_state.results
    display = st.session_state.display_results
    membership_df = st.session_state.membership_df

    display_out = display.reset_index()
    display_out.insert(0, "S.No.", range(1, len(display_out) + 1))
    display_out = display_out.rename(columns={"index": "Sequence_ID"})

    member_out = membership_df.copy()
    member_out.insert(0, "S.No.", range(1, len(member_out) + 1))

    tab1, tab2, tab3 = st.tabs([
        "Predictions",
        "Class Membership",
        "Summary"
    ])

    with tab1:
        st.dataframe(display_out, use_container_width=True, hide_index=True)

        csv = results.reset_index()
        csv.columns = [
            "Sequence_ID"
            if c == "index"
            else f"{c[0]}_{c[1]}"
            for c in csv.columns
        ]
        st.download_button(
            "Download full results (CSV)",
            csv.to_csv(index=False),
            "results.csv"
        )

    with tab2:
        st.dataframe(member_out, use_container_width=True, hide_index=True)
        st.download_button(
            "Download class membership (CSV)",
            member_out.to_csv(index=False),
            "class_membership.csv"
        )

    with tab3:
        total_valid = len(results)

        # Count assigned classes (multi-label)
        class_counts = pd.Series({
            cls: results[(cls, "Pred")].sum()
            for cls in CLASS_ORDER
        })

        # Count sequences with NO predicted class
        no_function_count = (
            results.loc[:, pd.IndexSlice[:, "Pred"]]
            .sum(axis=1)
            .eq(0)
            .sum()
        )

        # Combine counts
        class_counts["No_Function"] = no_function_count

        # Convert to percentages
        class_percentages = (
            class_counts / total_valid * 100
        ).round(2)

        # Remove zero bars
        class_percentages = class_percentages[class_percentages > 0]

        if not class_percentages.empty:

            df_plot = class_percentages.reset_index()
            df_plot.columns = ["Class", "Percentage"]

            # Add absolute counts for annotation
            df_plot["Count"] = df_plot["Class"].map(class_counts)

            fig = px.bar(
                df_plot,
                x="Class",
                y="Percentage",
                text=df_plot.apply(
                    lambda r: f"{r['Percentage']}% ({int(r['Count'])})",
                    axis=1
                ),
                title="Functional Assignment of Microproteins",
                color="Class",
                color_discrete_map=CLASS_COLORS
            )

            fig.update_traces(
                textposition="outside"
            )

            fig.update_layout(
                yaxis_title="Percentage of microproteins (%)",
                xaxis_title="Functional class",
                yaxis_range=[0, max(class_percentages.max() * 1.25, 5)],
                uniformtext_minsize=10,
                uniformtext_mode="hide"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Explicit textual summary (important for clarity)
            st.markdown(
                f"""
    **Summary**
    - Total valid microproteins analysed: **{total_valid}**
    - Microproteins with at least one predicted function: **{total_valid - no_function_count}**
    - Microproteins with **no predicted function**: **{no_function_count}**
                """
            )

        else:
            st.info("No sequences passed the prediction thresholds to generate class percentages.")

st.caption(
    "FAMILIE | Sequence-based microprotein classification using machine learning"
)