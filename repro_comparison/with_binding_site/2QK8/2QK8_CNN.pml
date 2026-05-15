# PyMOL Script - 2QK8 CNN
# Attention frequency (n=11 ligands)
# RED>75%  ORANGE 50-75%  YELLOW 25-50%
# BLUE = direct contact <=5A  MAGENTA = RED+contact
# Usage: File -> Run Script (keep pdb/sdf in same folder)

reinitialize
load 2qk8_protein.pdb, protein
load 2qk8_ligand.sdf,  ligand
hide everything, all
show cartoon, protein
color gray70, protein
show sticks, ligand
color cyan, ligand

select attn_red, (protein and resi 29+48+66+75+84+93+94+131+140+159)
color red, attn_red
show sticks, attn_red

select binding_site, (protein and resi 6+7+8+20+21+23+26+28+29+30+31+32+33+36+47+50+51+53+55+56+58+95+96+97+102+113+114+115+155)
color blue, binding_site
show sticks, binding_site

select consensus, attn_red and binding_site
color magenta, consensus
show sticks, consensus

zoom
bg_color white

print "=== 2QK8 CNN ==="
print "RED    (10): [29, 48, 66, 75, 84, 93, 94, 131, 140, 159]"
print "Contact (29): [6, 7, 8, 20, 21, 23, 26, 28, 29, 30, 31, 32, 33, 36, 47, 50, 51, 53, 55, 56, 58, 95, 96, 97, 102, 113, 114, 115, 155]"
