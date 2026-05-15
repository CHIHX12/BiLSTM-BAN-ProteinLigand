# PyMOL Script - 1AQ1 BiLSTM
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

select attn_red, (protein and resi 55+58+78+79+80+81+82+83+84+85+86+87+88+89+90+91+92+93+94+95+96+97+98+99+100+101+102+103+104+105+106+108+109+111+112+120+122+123+124+125+126+127+128+202+203+204+205+206+207)
color red, attn_red
show sticks, attn_red

select attn_orange, (protein and resi 56+57+59+67+114)
color orange, attn_orange
show sticks, attn_orange

select attn_yellow, (protein and resi 37+61+110+113+121+134+208)
color yellow, attn_yellow
show sticks, attn_yellow

select binding_site, (protein and resi 10+11+12+13+14+15+16+18+31+33+64+78+80+81+82+83+84+85+86+88+89+131+132+133+134+144+145+148)
color blue, binding_site
show sticks, binding_site

select consensus, attn_red and binding_site
color magenta, consensus
show sticks, consensus

zoom
bg_color white

print "=== 1AQ1 BiLSTM ==="
print "RED    (49): [55, 58, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 108, 109, 111, 112, 120, 122, 123, 124, 125, 126, 127, 128, 202, 203, 204, 205, 206, 207]"
print "Contact (28): [10, 11, 12, 13, 14, 15, 16, 18, 31, 33, 64, 78, 80, 81, 82, 83, 84, 85, 86, 88, 89, 131, 132, 133, 134, 144, 145, 148]"
