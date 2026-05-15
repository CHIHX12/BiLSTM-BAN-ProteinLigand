# PyMOL Script - 1QFS CNN + fpocket  [PREDICTED POCKET SPHERE]
#
# Residue colors:
#   MAGENTA : Consensus = CNN RED + fpocket  (weight 4)
#   RED     : CNN attention >75%  (weight 3)
#   ORANGE  : CNN attention 50-75%  (weight 2)
#   YELLOW  : CNN attention 25-50%  (weight 1)
#   BLUE    : fpocket only
#   CYAN    : Ligand (3D reference - NOT used for prediction)
#   GRAY    : Other residues
#
# Spheres:
#   MAGENTA sphere = consensus residues centroid (model RED + fpocket overlap)
#     -> guaranteed near the true pocket (highest confidence signal)
#   RED sphere(s) = DBSCAN clusters of all attention residues
#     -> shows full attention landscape (may include non-pocket regions)
#   BLUE sphere = fpocket reference center
#
# Interpretation:
#   MAGENTA sphere always near BLUE (by definition: both in pocket)
#   RED spheres show WHERE ELSE the model is attending
#   BiLSTM: MAGENTA sphere prominent + few gold clusters
#   CNN:    MAGENTA sphere small + many scattered red clusters
#
# NOTE: fpocket uses protein 3D structure only, no ligand needed.
#       Cyan ligand shown for spatial reference only.
#
# Usage: File -> Run Script  (keep pdb/sdf in same folder)

reinitialize
load 1qfs_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray80, protein

load 1qfs_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# MAGENTA: CNN RED + fpocket (8 residues, weight 4)
select consensus, (protein and resi 38+180+208+217+252+255+299+643)
color magenta, consensus
show sticks, consensus
# RED: CNN attention >75% (78 residues, weight 3)
select attn_only, (protein and resi 2+13+15+49+51+68+69+70+88+94+95+96+114+125+140+142+150+168+169+172+195+204+215+232+234+245+316+325+330+336+372+390+391+400+401+410+418+421+427+468+469+474+475+482+487+495+501+505+520+528+530+541+551+556+557+558+566+575+585+588+589+597+603+606+616+625+631+632+635+649+662+670+672+677+688+696+700+709)
color red, attn_only
show sticks, attn_only
# BLUE: fpocket only (51 residues)
select fpocket_only, (protein and resi 10+12+30+33+34+35+36+39+40+43+47+173+174+175+179+181+183+188+196+200+212+214+221+223+235+237+244+250+254+266+271+272+273+274+296+301+315+345+346+347+366+367+591+594+595+639+646+650+674+675+676)
color blue, fpocket_only
show sticks, fpocket_only
# ORANGE: attention 50-75% (8 residues, weight 2)
select attn_orange, (protein and resi 30+84+319+320+328+367+430+706)
color orange, attn_orange
show sticks, attn_orange
# YELLOW: attention 25-50% (3 residues, weight 1)
select attn_yellow, (protein and resi 502+569+604)
color yellow, attn_yellow
show sticks, attn_yellow

# Combined fpocket selection (for centroid calculation)
select fpocket_all, (protein and resi 10+12+30+33+34+35+36+38+39+40+43+47+173+174+175+179+180+181+183+188+196+200+208+212+214+217+221+223+235+237+244+250+252+254+255+266+271+272+273+274+296+299+301+315+345+346+347+366+367+591+594+595+639+643+646+650+674+675+676)

# ORANGE residues that also overlap fpocket (weight 2 for magenta sphere)
select orange_in_fp, (protein and resi 30+367)

# Inject constants before CGO block
python
MODEL_SEL_WEIGHTS     = {'consensus': 4, 'attn_only': 3, 'attn_orange': 2, 'attn_yellow': 1}
CONSENSUS_SEL_WEIGHTS = {'consensus': 3, 'orange_in_fp': 2}
MODEL_SPHERE_COLOR    = "tv_red"
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

print "=== 1QFS CNN Predicted Active Site (DBSCAN clusters) ==="
print "CNN attention: 86 residues -> RED sphere(s) per spatial cluster"
print "fpocket: 59 residues -> BLUE reference sphere"
print "BiLSTM: few compact clusters | CNN: many scattered clusters"
print "Nearest cluster to BLUE sphere = predicted binding site"
