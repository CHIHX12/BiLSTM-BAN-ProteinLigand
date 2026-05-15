# PyMOL Script - 5HLS CNN Attention (no binding site marked)
# Use case: unknown protein / AlphaFold3 structure
#
# Color scheme:
#   RED    : Attention > 75%  (most drugs consistently focus here)
#   ORANGE : Attention 50-75%
#   YELLOW : Attention 25-50%
#   GRAY   : Other residues
#
# NOTE: CNN attention is window-based (not true per-residue).
# Typically shows scattered RED with no clear cluster -> hard to interpret.
#
# Usage: File -> Run Script (keep 5hls_protein.pdb in same folder)

reinitialize
load 5hls_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 5hls_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# RED: attention > 75%  (27 residues)
select attn_red, (protein and resi 52+60+61+69+70+80+81+82+83+84+85+86+87+97+107+116+126+135+144+145+146+147+148+149+150+151+152)
color red, attn_red
show sticks, attn_red

zoom
bg_color white
