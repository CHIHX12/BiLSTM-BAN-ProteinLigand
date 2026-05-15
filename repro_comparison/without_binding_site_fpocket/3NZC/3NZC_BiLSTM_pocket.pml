# PyMOL Script - 3NZC BiLSTM + fpocket  [PREDICTED POCKET SPHERE]
#
# Residue colors:
#   MAGENTA : Consensus = BiLSTM RED + fpocket  (weight 4)
#   RED     : BiLSTM attention >75%  (weight 3)
#   ORANGE  : BiLSTM attention 50-75%  (weight 2)
#   YELLOW  : BiLSTM attention 25-50%  (weight 1)
#   BLUE    : fpocket only
#   CYAN    : Ligand (3D reference - NOT used for prediction)
#   GRAY    : Other residues
#
# Spheres:
#   MAGENTA sphere = consensus residues centroid (model RED + fpocket overlap)
#     -> guaranteed near the true pocket (highest confidence signal)
#   GOLD sphere(s) = DBSCAN clusters of all attention residues
#     -> shows full attention landscape (may include non-pocket regions)
#   BLUE sphere = fpocket reference center
#
# Interpretation:
#   MAGENTA sphere always near BLUE (by definition: both in pocket)
#   GOLD spheres show WHERE ELSE the model is attending
#   BiLSTM: MAGENTA sphere prominent + few gold clusters
#   CNN:    MAGENTA sphere small + many scattered red clusters
#
# NOTE: fpocket uses protein 3D structure only, no ligand needed.
#       Cyan ligand shown for spatial reference only.
#
# Usage: File -> Run Script  (keep pdb/sdf in same folder)

reinitialize
load 3nzc_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray80, protein

load 3nzc_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# MAGENTA: BiLSTM RED + fpocket (21 residues, weight 4)
select consensus, (protein and resi 19+20+21+23+24+25+26+27+28+31+32+33+34+125+126+127+128+129+131+134+135)
color magenta, consensus
show sticks, consensus
# RED: BiLSTM attention >75% (12 residues, weight 3)
select attn_only, (protein and resi 6+7+13+16+18+22+29+113+116+130+132+133)
color red, attn_only
show sticks, attn_only
# BLUE: fpocket only (48 residues)
select fpocket_only, (protein and resi 10+11+12+30+35+36+37+38+58+59+60+61+62+63+64+65+66+69+72+73+75+79+80+81+82+83+84+85+86+87+88+92+93+94+96+97+98+99+123+124+136+154+186+190+193+195+197+199)
color blue, fpocket_only
show sticks, fpocket_only
# ORANGE: attention 50-75% (8 residues, weight 2)
select attn_orange, (protein and resi 1+3+4+5+30+114+136+138)
color orange, attn_orange
show sticks, attn_orange
# YELLOW: attention 25-50% (5 residues, weight 1)
select attn_yellow, (protein and resi 31+32+33+34+125)
color yellow, attn_yellow
show sticks, attn_yellow

# Combined fpocket selection (for centroid calculation)
select fpocket_all, (protein and resi 10+11+12+19+20+21+23+24+25+26+27+28+30+31+32+33+34+35+36+37+38+58+59+60+61+62+63+64+65+66+69+72+73+75+79+80+81+82+83+84+85+86+87+88+92+93+94+96+97+98+99+123+124+125+126+127+128+129+131+134+135+136+154+186+190+193+195+197+199)

# ORANGE residues that also overlap fpocket (weight 2 for magenta sphere)
select orange_in_fp, (protein and resi 30+136)

# YELLOW residues that also overlap fpocket (weight 1 for magenta sphere)
select yellow_in_fp, (protein and resi 31+32+33+34+125)

# Inject constants before CGO block
python
MODEL_SEL_WEIGHTS     = {'consensus': 4, 'attn_only': 3, 'attn_orange': 2, 'attn_yellow': 1}
CONSENSUS_SEL_WEIGHTS = {'consensus': 3, 'orange_in_fp': 2, 'yellow_in_fp': 1}
MODEL_SPHERE_COLOR    = "gold"
POCKET_RADIUS         = 12.0
python end
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

zoom
bg_color white

print "=== 3NZC BiLSTM Predicted Active Site (DBSCAN clusters) ==="
print "BiLSTM attention: 33 residues -> GOLD sphere(s) per spatial cluster"
print "fpocket: 69 residues -> BLUE reference sphere"
print "BiLSTM: few compact clusters | CNN: many scattered clusters"
print "Nearest cluster to BLUE sphere = predicted binding site"
