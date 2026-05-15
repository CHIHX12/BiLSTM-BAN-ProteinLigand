#!/usr/bin/env python3
"""
generate_pocket_pmls.py
=======================
Generates {PDB}_{model}_pocket.pml for all 8 proteins.

Visualization:
  - Residue colors: MAGENTA/RED/ORANGE/YELLOW (same as before)
  - GOLD semi-transparent sphere  = model predicted active site
    (weighted centroid of colored residues: MAGENTA*4, RED*3, ORANGE*2, YELLOW*1)
  - BLUE semi-transparent sphere  = fpocket predicted pocket center
    (centroid of fpocket_only + consensus)
  - CYAN sticks = ligand (spatial reference)

Key insight:
  - BiLSTM sphere should land ON the actual pocket
  - CNN sphere may drift to protein interior (scattered residues)
  - Both spheres vs fpocket sphere = quantitative spatial comparison

Run from DrugBAN_BiLSTM root:
  python repro_comparison/generate_pocket_pmls.py
"""
import os, re

PROTEINS = ["1AQ1", "1OXG", "1QFS", "1T48", "2QK8", "3NZC", "5HLS", "6G3Q"]
MODELS   = ["BiLSTM", "CNN"]

BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "without_binding_site_fpocket"
)

# Weights for attention grades
WEIGHTS = {"consensus": 4, "attn_only": 3, "attn_orange": 2, "attn_yellow": 1}

POCKET_RADIUS = 12.0   # Angstrom - typical binding pocket radius

CGO_POCKET_BLOCK = '''\
python
from pymol import cmd
import numpy as _np

# -- helpers ------------------------------------------------------------------
def _collect(sel_weights):
    """Gather CA coords + per-atom weights from {selection: weight} dict."""
    coords, weights = [], []
    for sel, w in sel_weights.items():
        sp = {"c": []}
        cmd.iterate_state(1, f"({sel}) and name CA",
                          "c.append((x,y,z))", space=sp)
        for c in sp["c"]:
            coords.append(c)
            weights.append(w)
    if not coords:
        return None, None
    return _np.array(coords), _np.array(weights, dtype=float)

def _dbscan(pts, eps=8.0, min_pts=2):
    """Minimal DBSCAN (no sklearn needed). Returns label array (-1 = noise)."""
    n = len(pts)
    labels = [-1] * n
    cluster_id = 0
    visited = [False] * n

    def neighbors(i):
        return [j for j in range(n)
                if _np.linalg.norm(pts[i] - pts[j]) <= eps]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nb = neighbors(i)
        if len(nb) < min_pts:
            continue                      # noise for now
        labels[i] = cluster_id
        queue = list(nb)
        while queue:
            j = queue.pop()
            if not visited[j]:
                visited[j] = True
                nb2 = neighbors(j)
                if len(nb2) >= min_pts:
                    queue.extend(nb2)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1
    return _np.array(labels), cluster_id

def _sphere(name, xyz, radius, color_name, transp):
    cx, cy, cz = float(xyz[0]), float(xyz[1]), float(xyz[2])
    cmd.pseudoatom(name, pos=[cx, cy, cz])
    cmd.alter(name, f"vdw={radius:.2f}")
    cmd.rebuild(name)
    cmd.show("spheres", name)
    cmd.set("sphere_scale", 1.0, name)
    cmd.color(color_name, name)
    cmd.set("sphere_transparency", transp, name)

def draw_cluster_spheres(sel_weights, sphere_color, label_prefix):
    pts, wts = _collect(sel_weights)
    if pts is None:
        print(f"  [cluster] No residues for {label_prefix}")
        return []
    labels, n_clusters = _dbscan(pts, eps=8.0, min_pts=2)
    print(f"  [cluster] {label_prefix}: {len(pts)} residues -> "
          f"{n_clusters} cluster(s) + {(labels==-1).sum()} noise")
    centroids = []
    # Sort clusters by total weight descending (largest first)
    cluster_weights = []
    for cid in range(n_clusters):
        mask = labels == cid
        cluster_weights.append((cid, float(wts[mask].sum()), int(mask.sum())))
    cluster_weights.sort(key=lambda x: -x[1])

    max_w = cluster_weights[0][1] if cluster_weights else 1.0
    for rank, (cid, total_w, n_atoms) in enumerate(cluster_weights):
        mask = labels == cid
        ctr = _np.average(pts[mask], axis=0, weights=wts[mask])
        # Radius: scale with cluster atom count (min 6, max 14 A)
        r = min(14.0, max(6.0, 5.0 + n_atoms * 0.3))
        # Transparency: most prominent cluster most opaque
        transp = 0.40 + 0.25 * (rank / max(1, n_clusters - 1))
        transp = min(0.75, transp)
        obj_name = f"{label_prefix}_c{rank+1}"
        _sphere(obj_name, ctr, r, sphere_color, transp)
        print(f"    cluster {rank+1}: {n_atoms} residues, "
              f"weight={total_w:.0f}, r={r:.1f}A, "
              f"center=({ctr[0]:.1f},{ctr[1]:.1f},{ctr[2]:.1f})")
        centroids.append((ctr, total_w))
    # Handle noise atoms as individual tiny markers
    noise_mask = labels == -1
    if noise_mask.any():
        print(f"    ({noise_mask.sum()} noise residues, no sphere drawn)")
    return centroids

# -- CONSENSUS sphere (all attention grades that overlap fpocket) --------------
# RED(weight 3) + ORANGE(weight 2) + YELLOW(weight 1) that are in fpocket
# CONSENSUS_SEL_WEIGHTS injected from PML constants block
con_pts, con_wts = _collect(CONSENSUS_SEL_WEIGHTS)
if con_pts is not None and len(con_pts) >= 1:
    con_ctr = _np.average(con_pts, axis=0, weights=con_wts)
    n_con   = len(con_pts)
    # Radius scales with total count: min 5A, max 12A
    con_r = min(12.0, max(5.0, 4.0 + n_con * 0.4))
    _sphere("consensus_site", con_ctr, con_r, "magenta", 0.40)
    grades = " + ".join(f"{k}*{v}" for k, v in CONSENSUS_SEL_WEIGHTS.items())
    print(f"  [consensus] {n_con} residues ({grades})")
    print(f"    -> MAGENTA sphere r={con_r:.1f}A "
          f"center=({con_ctr[0]:.1f},{con_ctr[1]:.1f},{con_ctr[2]:.1f})")
else:
    print(f"  [consensus] No attention residues overlap fpocket")

# -- MODEL attention clusters (gold/red) for broader attention landscape ------
model_centers = draw_cluster_spheres(
    MODEL_SEL_WEIGHTS, MODEL_SPHERE_COLOR, "pred")

# -- FPOCKET single centroid sphere (reference) --------------------------------
fp_pts, fp_wts = _collect({"fpocket_all": 1})
if fp_pts is not None:
    fp_ctr = _np.average(fp_pts, axis=0)
    _sphere("fpocket_ref", fp_ctr, POCKET_RADIUS, "blue", 0.70)
    print(f"  [fpocket] center=({fp_ctr[0]:.1f},{fp_ctr[1]:.1f},{fp_ctr[2]:.1f})")
    # Distance: consensus sphere to fpocket (should be small by definition)
    if con_pts is not None and len(con_pts) > 0:
        con_ctr2 = _np.average(con_pts, axis=0, weights=con_wts)
        d_con = float(_np.linalg.norm(con_ctr2 - fp_ctr))
        print(f"  [fpocket] MAGENTA to BLUE distance: {d_con:.1f} A "
              f"({'OVERLAP' if d_con < POCKET_RADIUS else 'no overlap'})")
    # Distance: nearest attention cluster to fpocket
    if model_centers:
        dists = [(_np.linalg.norm(c - fp_ctr), w) for c, w in model_centers]
        best_d, _ = min(dists, key=lambda x: x[0])
        print(f"  [fpocket] Nearest attention cluster to BLUE: {best_d:.1f} A")

python end
'''


def parse_resi(line):
    m = re.search(r'resi\s+([\d+]+)', line)
    if not m:
        return []
    return [int(x) for x in m.group(1).split("+")]

def resi_str(ids):
    return "+".join(str(i) for i in sorted(set(ids)))

def generate_pocket_pml(pdb: str, model: str):
    pdb_lo = pdb.lower()
    src_dir = os.path.join(BASE, pdb)
    src_pml = os.path.join(src_dir, f"{pdb}_{model}_fpocket.pml")

    if not os.path.exists(src_pml):
        print(f"  [skip] {src_pml} not found")
        return

    with open(src_pml) as f:
        lines = f.readlines()

    sel = {"consensus": [], "attn_only": [], "fpocket_only": [],
           "attn_orange": [], "attn_yellow": []}
    for line in lines:
        for key in sel:
            if line.strip().startswith(f"select {key},"):
                sel[key] = parse_resi(line)

    model_red  = sorted(set(sel["consensus"] + sel["attn_only"]))
    fpocket    = sorted(set(sel["consensus"] + sel["fpocket_only"]))

    # Sphere color: gold for BiLSTM, tomato red for CNN
    sphere_color = "gold" if model == "BiLSTM" else "tv_red"
    sphere_label = "GOLD" if model == "BiLSTM" else "RED"

    # Build MODEL_SEL_WEIGHTS as a Python dict literal for the CGO block
    present_sels = {}
    if sel["consensus"]:   present_sels["consensus"]  = 4
    if sel["attn_only"]:   present_sels["attn_only"]  = 3
    if sel["attn_orange"]: present_sels["attn_orange"]= 2
    if sel["attn_yellow"]: present_sels["attn_yellow"]= 1

    n_attention = len(model_red)
    n_fpocket   = len(fpocket)

    header = f"""\
# PyMOL Script - {pdb} {model} + fpocket  [PREDICTED POCKET SPHERE]
#
# Residue colors:
#   MAGENTA : Consensus = {model} RED + fpocket  (weight 4)
#   RED     : {model} attention >75%  (weight 3)
#   ORANGE  : {model} attention 50-75%  (weight 2)
#   YELLOW  : {model} attention 25-50%  (weight 1)
#   BLUE    : fpocket only
#   CYAN    : Ligand (3D reference - NOT used for prediction)
#   GRAY    : Other residues
#
# Spheres:
#   MAGENTA sphere = consensus residues centroid (model RED + fpocket overlap)
#     -> guaranteed near the true pocket (highest confidence signal)
#   {sphere_label} sphere(s) = DBSCAN clusters of all attention residues
#     -> shows full attention landscape (may include non-pocket regions)
#   BLUE sphere = fpocket reference center
#
# Interpretation:
#   MAGENTA sphere always near BLUE (by definition: both in pocket)
#   {sphere_label} spheres show WHERE ELSE the model is attending
#   BiLSTM: MAGENTA sphere prominent + few gold clusters
#   CNN:    MAGENTA sphere small + many scattered red clusters
#
# NOTE: fpocket uses protein 3D structure only, no ligand needed.
#       Cyan ligand shown for spatial reference only.
#
# Usage: File -> Run Script  (keep pdb/sdf in same folder)

reinitialize
load {pdb_lo}_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray80, protein

load {pdb_lo}_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand
"""

    def sel_block(name, ids, color, note=""):
        if not ids:
            return f"# (no {name})\n"
        lines_out  = f"# {note}\n" if note else ""
        lines_out += f"select {name}, (protein and resi {resi_str(ids)})\n"
        lines_out += f"color {color}, {name}\n"
        lines_out += f"show sticks, {name}\n"
        return lines_out

    body  = sel_block("consensus",    sel["consensus"],    "magenta",
                      f"MAGENTA: {model} RED + fpocket ({len(sel['consensus'])} residues, weight 4)")
    body += sel_block("attn_only",    sel["attn_only"],    "red",
                      f"RED: {model} attention >75% ({len(sel['attn_only'])} residues, weight 3)")
    body += sel_block("fpocket_only", sel["fpocket_only"], "blue",
                      f"BLUE: fpocket only ({len(sel['fpocket_only'])} residues)")
    if sel["attn_orange"]:
        body += sel_block("attn_orange", sel["attn_orange"], "orange",
                          f"ORANGE: attention 50-75% ({len(sel['attn_orange'])} residues, weight 2)")
    if sel["attn_yellow"]:
        body += sel_block("attn_yellow", sel["attn_yellow"], "yellow",
                          f"YELLOW: attention 25-50% ({len(sel['attn_yellow'])} residues, weight 1)")

    # Compute orange/yellow residues that also overlap fpocket
    orange_fp = sorted(set(sel["attn_orange"]) & set(fpocket))
    yellow_fp = sorted(set(sel["attn_yellow"]) & set(fpocket))

    # Consensus weights: all attention grades that are IN fpocket
    consensus_sel_weights = {}
    if sel["consensus"]:  consensus_sel_weights["consensus"]    = 3
    if orange_fp:         consensus_sel_weights["orange_in_fp"] = 2
    if yellow_fp:         consensus_sel_weights["yellow_in_fp"] = 1
    n_con_total = len(sel["consensus"]) + len(orange_fp) + len(yellow_fp)

    combined = f"""
# Combined fpocket selection (for centroid calculation)
select fpocket_all, (protein and resi {resi_str(fpocket) if fpocket else "none"})
"""
    if orange_fp:
        combined += f"""
# ORANGE residues that also overlap fpocket (weight 2 for magenta sphere)
select orange_in_fp, (protein and resi {resi_str(orange_fp)})
"""
    if yellow_fp:
        combined += f"""
# YELLOW residues that also overlap fpocket (weight 1 for magenta sphere)
select yellow_in_fp, (protein and resi {resi_str(yellow_fp)})
"""
    combined += f"""
# Inject constants before CGO block
python
MODEL_SEL_WEIGHTS     = {present_sels!r}
CONSENSUS_SEL_WEIGHTS = {consensus_sel_weights!r}
MODEL_SPHERE_COLOR    = "{sphere_color}"
POCKET_RADIUS         = {POCKET_RADIUS}
python end
"""

    footer = f"""
zoom
bg_color white

print "=== {pdb} {model} Predicted Active Site (DBSCAN clusters) ==="
print "{model} attention: {n_attention} residues -> {sphere_label} sphere(s) per spatial cluster"
print "fpocket: {n_fpocket} residues -> BLUE reference sphere"
print "BiLSTM: few compact clusters | CNN: many scattered clusters"
print "Nearest cluster to BLUE sphere = predicted binding site"
"""

    content = header + "\n" + body + combined + CGO_POCKET_BLOCK + footer

    out_path = os.path.join(src_dir, f"{pdb}_{model}_pocket.pml")
    with open(out_path, "w") as f:
        f.write(content)
    print(f"  Written: {out_path}  (attn={n_attention}, fpocket={n_fpocket})")


if __name__ == "__main__":
    for pdb in PROTEINS:
        print(f"\n[{pdb}]")
        for model in MODELS:
            generate_pocket_pml(pdb, model)

    print("\nDone.")
    print("In PyMOL: File -> Run Script -> *_pocket.pml")
    print("  GOLD sphere  = BiLSTM predicted active site")
    print("  RED sphere   = CNN predicted active site")
    print("  BLUE sphere  = fpocket reference")
    print("  Overlap between GOLD and BLUE = BiLSTM agrees with fpocket!")
