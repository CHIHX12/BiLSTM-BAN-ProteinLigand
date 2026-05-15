#!/usr/bin/env python3
"""
generate_bbox_pmls.py
=====================
Reads existing without_binding_site_fpocket/{PDB}/{PDB}_{model}_fpocket.pml
and writes new {PDB}_{model}_bbox.pml files that add CGO axis-aligned
bounding boxes around:
  - MODEL_RED (consensus + attn_only)   → red wireframe box
  - FPOCKET   (consensus + fpocket_only) → blue wireframe box

The box volumes make spatial coherence immediately visible:
  - BiLSTM RED → compact box = coherent binding pocket
  - CNN RED    → large scattered box = no spatial clustering

Run from any directory:
  python repro_comparison/generate_bbox_pmls.py
"""
import os, re

PROTEINS = ["1AQ1", "1OXG", "1QFS", "1T48", "2QK8", "3NZC", "5HLS", "6G3Q"]
MODELS   = ["BiLSTM", "CNN"]

BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "without_binding_site_fpocket"
)

# ── CGO bounding box Python block (injected into PML) ──────────────────────
# PyMOL scripting: "python" ... "python end" blocks run inline Python.
# We build the box from CA atoms of the selection.
CGO_BLOCK = '''
python
from pymol import cmd
from pymol.cgo import LINEWIDTH, BEGIN, LINES, COLOR, VERTEX, END

def _draw_bbox(selection, rgb, cgo_name, padding=2.5):
    """Draw axis-aligned bounding box around CA atoms of a selection."""
    import numpy as np
    try:
        space = {"coords": []}
        cmd.iterate_state(1, f"({selection}) and name CA",
                          "coords.append((x,y,z))", space=space)
        if not space["coords"]:
            print(f"  [bbox] no CA atoms in '{selection}', skipped")
            return
        import numpy as _np
        pts = _np.array(space["coords"])
        mn = pts.min(0) - padding
        mx = pts.max(0) + padding
        x0,y0,z0 = mn; x1,y1,z1 = mx
        r,g,b = rgb
        vol = (x1-x0)*(y1-y0)*(z1-z0)
        print(f"  [bbox] '{cgo_name}'  {len(pts)} CA atoms  vol={vol:.0f} A^3")
        box = [LINEWIDTH, 2.5, BEGIN, LINES, COLOR, r,g,b,
               VERTEX,x0,y0,z0, VERTEX,x1,y0,z0,
               VERTEX,x1,y0,z0, VERTEX,x1,y1,z0,
               VERTEX,x1,y1,z0, VERTEX,x0,y1,z0,
               VERTEX,x0,y1,z0, VERTEX,x0,y0,z0,
               VERTEX,x0,y0,z1, VERTEX,x1,y0,z1,
               VERTEX,x1,y0,z1, VERTEX,x1,y1,z1,
               VERTEX,x1,y1,z1, VERTEX,x0,y1,z1,
               VERTEX,x0,y1,z1, VERTEX,x0,y0,z1,
               VERTEX,x0,y0,z0, VERTEX,x0,y0,z1,
               VERTEX,x1,y0,z0, VERTEX,x1,y0,z1,
               VERTEX,x1,y1,z0, VERTEX,x1,y1,z1,
               VERTEX,x0,y1,z0, VERTEX,x0,y1,z1,
               END]
        cmd.load_cgo(box, cgo_name)
    except Exception as e:
        print(f"  [bbox] error for '{cgo_name}': {e}")

# model RED = consensus + attn_only  (combined as 'model_red' selection)
# fpocket   = consensus + fpocket_only (combined as 'fpocket_all' selection)
_draw_bbox("model_red",   MODEL_RED_RGB,   "box_model_red",  padding=2.5)
_draw_bbox("fpocket_all", (0.1, 0.4, 1.0), "box_fpocket",    padding=2.5)

python end
'''

def parse_resi(line):
    """Extract residue IDs from a PyMOL 'select ..., (protein and resi ...)' line."""
    m = re.search(r'resi\s+([\d+]+)', line)
    if not m:
        return []
    return [int(x) for x in m.group(1).split("+")]

def resi_str(ids):
    return "+".join(str(i) for i in sorted(set(ids)))

def generate_bbox_pml(pdb: str, model: str):
    pdb_lo = pdb.lower()
    src_dir = os.path.join(BASE, pdb)
    src_pml = os.path.join(src_dir, f"{pdb}_{model}_fpocket.pml")

    if not os.path.exists(src_pml):
        print(f"  [skip] {src_pml} not found")
        return

    with open(src_pml) as f:
        lines = f.readlines()

    # Parse selections
    sel = {"consensus": [], "attn_only": [], "fpocket_only": [],
           "attn_orange": [], "attn_yellow": []}
    for line in lines:
        for key in sel:
            if line.strip().startswith(f"select {key},"):
            #if f"select {key}," in line:
                sel[key] = parse_resi(line)

    model_red = sorted(set(sel["consensus"] + sel["attn_only"]))
    fpocket   = sorted(set(sel["consensus"] + sel["fpocket_only"]))

    # Color for model RED box: red for CNN, orangered for BiLSTM
    if model == "CNN":
        red_rgb_str = "(1.0, 0.1, 0.1)"
        red_rgb_label = "red"
    else:
        red_rgb_str = "(1.0, 0.4, 0.0)"
        red_rgb_label = "orangered"

    header = f"""\
# PyMOL Script - {pdb} {model} + fpocket  [WITH BOUNDING BOXES]
#
# Color scheme (residues):
#   MAGENTA : Consensus = {model} RED and fpocket  (strongest signal)
#   RED     : {model} attention only (>75%, not in fpocket)
#   BLUE    : fpocket only (not in {model} RED)
#   ORANGE  : {model} attention 50-75%
#   YELLOW  : {model} attention 25-50%
#   CYAN    : Ligand (spatial reference only)
#   GRAY    : Other residues
#
# Bounding boxes (CGO wireframe):
#   {red_rgb_label.upper()} box : {model} RED residues ({len(model_red)} total)
#   BLUE box       : fpocket residues ({len(fpocket)} total)
#
# Interpretation:
#   Compact box = spatially clustered = coherent binding pocket
#   Large box   = scattered = low spatial coherence
#
# NOTE: fpocket requires only the protein PDB (no ligand needed).
#       The cyan ligand here is for 3D spatial reference only.
#
# Usage: File -> Run Script  (keep pdb/sdf in same folder)

reinitialize
load {pdb_lo}_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load {pdb_lo}_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# MAGENTA: {model} RED + fpocket consensus ({len(sel['consensus'])} residues)
"""
    def select_block(name, ids, color_name, show=True):
        if not ids:
            return f"# (no {name} residues)\n"
        s  = f"select {name}, (protein and resi {resi_str(ids)})\n"
        s += f"color {color_name}, {name}\n"
        if show:
            s += f"show sticks, {name}\n"
        return s

    body  = select_block("consensus",   sel["consensus"],   "magenta")
    body += f"\n# RED: {model} attention only ({len(sel['attn_only'])} residues)\n"
    body += select_block("attn_only",   sel["attn_only"],   "red")
    body += f"\n# BLUE: fpocket only ({len(sel['fpocket_only'])} residues)\n"
    body += select_block("fpocket_only",sel["fpocket_only"],"blue")

    if sel["attn_orange"]:
        body += f"\n# ORANGE: attention 50-75% ({len(sel['attn_orange'])} residues)\n"
        body += select_block("attn_orange", sel["attn_orange"], "orange")
    if sel["attn_yellow"]:
        body += f"\n# YELLOW: attention 25-50% ({len(sel['attn_yellow'])} residues)\n"
        body += select_block("attn_yellow", sel["attn_yellow"], "yellow")

    # Combined selections for bounding boxes
    combined = f"""
# ── Combined selections for bounding boxes ──────────────────────────────────
# model_red = consensus + attn_only
select model_red, (protein and resi {resi_str(model_red) if model_red else "none"})

# fpocket_all = consensus + fpocket_only
select fpocket_all, (protein and resi {resi_str(fpocket) if fpocket else "none"})

# Inject MODEL_RED_RGB before the CGO block
python
MODEL_RED_RGB = {red_rgb_str}
python end
"""

    footer = f"""
zoom
bg_color white
set line_width, 2

print "=== {pdb} {model}+fpocket  [bbox] ==="
print "{model} RED ({len(model_red)} residues) -> {red_rgb_label} wireframe box"
print "fpocket ({len(fpocket)} residues) -> blue wireframe box"
print "Compare box volumes to judge spatial coherence!"
"""

    content = header + body + combined + CGO_BLOCK + footer

    out_path = os.path.join(src_dir, f"{pdb}_{model}_bbox.pml")
    with open(out_path, "w") as f:
        f.write(content)
    print(f"  Written: {out_path}")
    print(f"    {model} RED: {len(model_red)} residues | fpocket: {len(fpocket)} residues")


if __name__ == "__main__":
    for pdb in PROTEINS:
        print(f"\n[{pdb}]")
        for model in MODELS:
            generate_bbox_pml(pdb, model)
    print("\nDone. Open *_bbox.pml in PyMOL (File -> Run Script).")
    print("Rotate to compare the RED box (model attention) vs BLUE box (fpocket).")
