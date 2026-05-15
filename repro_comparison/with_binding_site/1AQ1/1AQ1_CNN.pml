# PyMOL Script - 1AQ1 CNN
# Attention frequency (n=160 ligands)
# RED>75%  ORANGE 50-75%  YELLOW 25-50%
# BLUE = direct contact <=5A  MAGENTA = RED+contact
# Usage: File -> Run Script (keep pdb/sdf in same folder)

reinitialize
load 1aq1_protein.pdb, protein
load 1aq1_ligand.sdf,  ligand
hide everything, all
show cartoon, protein
color gray70, protein
show sticks, ligand
color cyan, ligand

select attn_red, (protein and resi 10+12+19+20+21+40+48+57+67+75+84+94+104+122+123+131+141+149+151+159+168+177+187+205+214+215+223+225+232+242+297)
color red, attn_red
show sticks, attn_red

select attn_orange, (protein and resi 93)
color orange, attn_orange
show sticks, attn_orange

select binding_site, (protein and resi 10+11+12+13+14+15+16+18+31+33+64+78+80+81+82+83+84+85+86+88+89+131+132+133+134+144+145+148)
color blue, binding_site
show sticks, binding_site

select consensus, attn_red and binding_site
color magenta, consensus
show sticks, consensus

zoom
bg_color white

print "=== 1AQ1 CNN ==="
print "RED    (31): [10, 12, 19, 20, 21, 40, 48, 57, 67, 75, 84, 94, 104, 122, 123, 131, 141, 149, 151, 159, 168, 177, 187, 205, 214, 215, 223, 225, 232, 242, 297]"
print "Contact (28): [10, 11, 12, 13, 14, 15, 16, 18, 31, 33, 64, 78, 80, 81, 82, 83, 84, 85, 86, 88, 89, 131, 132, 133, 134, 144, 145, 148]"
