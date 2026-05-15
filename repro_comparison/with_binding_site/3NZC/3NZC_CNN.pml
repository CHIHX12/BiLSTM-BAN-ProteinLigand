# PyMOL Script - 3NZC CNN
# Attention frequency (n=7 ligands)
# RED>75%  ORANGE 50-75%  YELLOW 25-50%
# BLUE = direct contact <=5A  MAGENTA = RED+contact
# Usage: File -> Run Script (keep pdb/sdf in same folder)

reinitialize
load 3nzc_protein.pdb, protein
load 3nzc_ligand.sdf,  ligand
hide everything, all
show cartoon, protein
color gray70, protein
show sticks, ligand
color cyan, ligand

select attn_red, (protein and resi 19+20+29+30+47+48+93+104+112+132+158+168+176+178+195+204+205+206)
color red, attn_red
show sticks, attn_red

select binding_site, (protein and resi 10+11+12+24+25+30+32+33+34+35+36+37+61+64+65+66+68+69+72+75+122+123+124+129+142+143+144+199)
color blue, binding_site
show sticks, binding_site

select consensus, attn_red and binding_site
color magenta, consensus
show sticks, consensus

zoom
bg_color white

print "=== 3NZC CNN ==="
print "RED    (18): [19, 20, 29, 30, 47, 48, 93, 104, 112, 132, 158, 168, 176, 178, 195, 204, 205, 206]"
print "Contact (28): [10, 11, 12, 24, 25, 30, 32, 33, 34, 35, 36, 37, 61, 64, 65, 66, 68, 69, 72, 75, 122, 123, 124, 129, 142, 143, 144, 199]"
