# PyMOL Script - 6G3Q CNN
# Attention frequency (n=26 ligands)
# RED>75%  ORANGE 50-75%  YELLOW 25-50%
# BLUE = direct contact <=5A  MAGENTA = RED+contact
# Usage: File -> Run Script (keep pdb/sdf in same folder)

reinitialize
load 6g3q_protein.pdb, protein
load 6g3q_ligand.sdf,  ligand
hide everything, all
show cartoon, protein
color gray70, protein
show sticks, ligand
color cyan, ligand

select attn_red, (protein and resi 10+19+30+39+56+58+66+67+75+103+104+113+151+160+178+179+198+206+215+234)
color red, attn_red
show sticks, attn_red

select attn_orange, (protein and resi 65+141+233+244)
color orange, attn_orange
show sticks, attn_orange

select attn_yellow, (protein and resi 85+112)
color yellow, attn_yellow
show sticks, attn_yellow

select binding_site, (protein and resi 92+94+96+106+119+121+131+132+135+141+143+197+198+199+200+201+202+203+204+207+209)
color blue, binding_site
show sticks, binding_site

select consensus, attn_red and binding_site
color magenta, consensus
show sticks, consensus

zoom
bg_color white

print "=== 6G3Q CNN ==="
print "RED    (20): [10, 19, 30, 39, 56, 58, 66, 67, 75, 103, 104, 113, 151, 160, 178, 179, 198, 206, 215, 234]"
print "Contact (21): [92, 94, 96, 106, 119, 121, 131, 132, 135, 141, 143, 197, 198, 199, 200, 201, 202, 203, 204, 207, 209]"
