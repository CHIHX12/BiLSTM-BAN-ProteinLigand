# PyMOL Script - 6G3Q BiLSTM
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

select attn_red, (protein and resi 6+7+8+9+10+11+12+13+14+15+16+17+18+98+99+100+101+102+103+104+105+144+145+148+151+152)
color red, attn_red
show sticks, attn_red

select attn_orange, (protein and resi 2+19+29+82+164+185)
color orange, attn_orange
show sticks, attn_orange

select attn_yellow, (protein and resi 1+5+20+28+81+83+84+86+146+147+149+150+156+165)
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

print "=== 6G3Q BiLSTM ==="
print "RED    (26): [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 98, 99, 100, 101, 102, 103, 104, 105, 144, 145, 148, 151, 152]"
print "Contact (21): [92, 94, 96, 106, 119, 121, 131, 132, 135, 141, 143, 197, 198, 199, 200, 201, 202, 203, 204, 207, 209]"
