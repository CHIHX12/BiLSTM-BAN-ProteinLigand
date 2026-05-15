"""
Model Benchmark Test
=====================
Loads the trained BiLSTM and CNN models and verifies that their AUROC on a
random subset of the BindingDB test set meets the published threshold.

Published results:
  BiLSTM (epoch 94): AUROC=0.9641, AUPRC=0.9542
  CNN    (epoch 90): AUROC=0.9603, AUPRC=0.9485

This test ensures:
  1. Checkpoint files are intact (not corrupted/missing).
  2. Model loads without error.
  3. On a 200-sample subset: AUROC ≥ 0.90  (conservative lower bound).
  4. Score distribution is sensible (not all zeros, not all ones).

Note: Uses a random seed to pick the same 200 samples every run.
"""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # GCN-BILSTM-BAN-bak
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))        # screening/

import pytest
import numpy as np
import pandas as pd
import torch
from functools import partial

# ─── Skip if model checkpoints missing ────────────────────────────────────
_PARENT = os.path.join(os.path.dirname(__file__), "..", "..")
_BILSTM_CKPT = os.path.join(_PARENT, "result", "DrugBAN_BiLSTM", "best_model_epoch_94.pth")
_CNN_CKPT    = os.path.join(_PARENT, "result", "DrugBAN",         "best_model_epoch_90.pth")
_TEST_CSV    = os.path.join(_PARENT, "datasets", "random", "test.csv")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(_BILSTM_CKPT),
    reason="BiLSTM checkpoint not found — skipping model benchmark"
)

SUBSET_SIZE = 200   # samples to evaluate (balance speed vs. reliability)
RANDOM_SEED = 42
MIN_AUROC   = 0.90  # conservative lower bound for the benchmark


def _load_model(model_type: str, device: torch.device):
    from models import DrugBAN
    from configs import get_cfg_defaults

    cfg_map = {
        "bilstm": ("DrugBAN_BiLSTM.yaml", _BILSTM_CKPT),
        "cnn":    ("DrugBAN.yaml",         _CNN_CKPT),
    }
    yaml_name, ckpt = cfg_map[model_type]
    yaml_path = os.path.join(_PARENT, "configs", yaml_name)

    cfg = get_cfg_defaults()
    if os.path.isfile(yaml_path):
        cfg.merge_from_file(yaml_path)

    model = DrugBAN(**cfg)
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model, cfg


def _build_drug_graph(smiles):
    from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
    atom_f = CanonicalAtomFeaturizer()
    bond_f = CanonicalBondFeaturizer(self_loop=True)
    fc = partial(smiles_to_bigraph, add_self_loop=True)
    MAX = 290
    g = fc(smiles=smiles, node_featurizer=atom_f, edge_featurizer=bond_f)
    if g is None:
        return None
    feats = g.ndata.pop("h")
    n = feats.shape[0]
    if n > MAX:
        return None
    feats = torch.cat([feats, torch.zeros(n, 1)], 1)
    g.ndata["h"] = feats
    virt = MAX - n
    g.add_nodes(virt, {"h": torch.cat([torch.zeros(virt, 74), torch.ones(virt, 1)], 1)})
    return g.add_self_loop()


def _run_inference(model, cfg, df_subset, device, use_bilstm):
    import dgl
    from utils import integer_label_protein, encode_sequence_with_features

    MAX_PROT = 1200
    probs, labels = [], []

    with torch.no_grad():
        for _, row in df_subset.iterrows():
            smiles = row["SMILES"]
            seq    = row["Protein"]
            label  = int(row["Y"])

            g = _build_drug_graph(smiles)
            if g is None:
                continue

            bg_d = dgl.batch([g]).to(device)

            if use_bilstm:
                from utils import encode_sequence_with_features
                p_idx, p_feat, p_len = encode_sequence_with_features(seq, max_length=MAX_PROT)
                v_p = (
                    torch.LongTensor(p_idx).unsqueeze(0).to(device),
                    torch.FloatTensor(p_feat).unsqueeze(0).to(device),
                    torch.LongTensor([p_len]).to(device),
                )
            else:
                from utils import integer_label_protein
                p = integer_label_protein(seq, max_length=MAX_PROT)
                v_p = torch.LongTensor(p).unsqueeze(0).to(device)

            _, _, score, _ = model(bg_d, v_p, mode="eval")

            n_class = cfg["DECODER"]["BINARY"]
            if n_class == 1:
                prob = float(torch.sigmoid(score).item())
            else:
                prob = float(torch.softmax(score, dim=1)[0, 1].item())

            probs.append(prob)
            labels.append(label)

    return np.array(probs), np.array(labels)


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def df_subset():
    assert os.path.isfile(_TEST_CSV), f"Test CSV not found: {_TEST_CSV}"
    df = pd.read_csv(_TEST_CSV)
    rng = random.Random(RANDOM_SEED)
    idx = rng.sample(range(len(df)), min(SUBSET_SIZE, len(df)))
    return df.iloc[idx].reset_index(drop=True)


# ─── BiLSTM checkpoint ────────────────────────────────────────────────────

class TestBiLSTMCheckpoint:
    def test_checkpoint_exists(self):
        assert os.path.isfile(_BILSTM_CKPT), f"Missing: {_BILSTM_CKPT}"

    def test_checkpoint_not_empty(self):
        size = os.path.getsize(_BILSTM_CKPT)
        assert size > 1_000_000, f"Checkpoint suspiciously small: {size} bytes"

    def test_model_loads(self, device):
        model, cfg = _load_model("bilstm", device)
        assert model is not None
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 1_000_000, f"Model has unexpectedly few params: {n_params}"

    def test_auroc_above_threshold(self, device, df_subset):
        from sklearn.metrics import roc_auc_score
        model, cfg = _load_model("bilstm", device)
        probs, labels = _run_inference(model, cfg, df_subset, device, use_bilstm=True)

        assert len(probs) >= 50, "Too few valid samples — check input data"
        auroc = roc_auc_score(labels, probs)
        print(f"\n  BiLSTM AUROC on {len(probs)}-sample subset: {auroc:.4f}")
        assert auroc >= MIN_AUROC, (
            f"AUROC {auroc:.4f} < {MIN_AUROC}. "
            "Model may be corrupted or loaded with wrong config."
        )

    def test_score_distribution_healthy(self, device, df_subset):
        """Scores should span a range — not all collapsing to 0 or 1."""
        model, cfg = _load_model("bilstm", device)
        probs, labels = _run_inference(model, cfg, df_subset, device, use_bilstm=True)

        assert probs.std() > 0.05, "Score variance too low — model may be degenerate"
        assert probs.min() < 0.3,  "All scores very high — possible sigmoid overflow"
        assert probs.max() > 0.7,  "All scores very low — model not predicting positives"

    def test_positive_class_scores_higher(self, device, df_subset):
        """Positive samples (Y=1) should score higher on average than negatives (Y=0)."""
        model, cfg = _load_model("bilstm", device)
        probs, labels = _run_inference(model, cfg, df_subset, device, use_bilstm=True)

        pos_mean = probs[labels == 1].mean() if (labels == 1).sum() > 0 else 0.5
        neg_mean = probs[labels == 0].mean() if (labels == 0).sum() > 0 else 0.5
        print(f"\n  Positive mean score: {pos_mean:.4f}, Negative mean score: {neg_mean:.4f}")
        assert pos_mean > neg_mean, (
            "Positive class does not score higher than negative — "
            "model predictions may be inverted or random."
        )


# ─── CNN checkpoint ───────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.path.isfile(_CNN_CKPT),
    reason="CNN checkpoint not found"
)
class TestCNNCheckpoint:
    def test_checkpoint_exists(self):
        assert os.path.isfile(_CNN_CKPT)

    def test_model_loads(self, device):
        model, cfg = _load_model("cnn", device)
        assert model is not None

    def test_auroc_above_threshold(self, device, df_subset):
        from sklearn.metrics import roc_auc_score
        model, cfg = _load_model("cnn", device)
        probs, labels = _run_inference(model, cfg, df_subset, device, use_bilstm=False)
        auroc = roc_auc_score(labels, probs)
        print(f"\n  CNN AUROC on {len(probs)}-sample subset: {auroc:.4f}")
        assert auroc >= MIN_AUROC
