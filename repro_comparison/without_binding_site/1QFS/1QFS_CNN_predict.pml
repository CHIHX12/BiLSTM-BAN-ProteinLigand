# PyMOL Script - 1QFS CNN Attention (no binding site marked)
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
# Usage: File -> Run Script (keep 1qfs_protein.pdb in same folder)

reinitialize
load 1qfs_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 1qfs_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# RED: attention > 75%  (86 residues)
select attn_red, (protein and resi 2+13+15+38+49+51+68+69+70+88+94+95+96+114+125+140+142+150+168+169+172+180+195+204+208+215+217+232+234+245+252+255+299+316+325+330+336+372+390+391+400+401+410+418+421+427+468+469+474+475+482+487+495+501+505+520+528+530+541+551+556+557+558+566+575+585+588+589+597+603+606+616+625+631+632+635+643+649+662+670+672+677+688+696+700+709)
color red, attn_red
show sticks, attn_red

# ORANGE: attention 50-75%  (8 residues)
select attn_orange, (protein and resi 30+84+319+320+328+367+430+706)
color orange, attn_orange
show sticks, attn_orange

# YELLOW: attention 25-50%  (3 residues)
select attn_yellow, (protein and resi 502+569+604)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white
