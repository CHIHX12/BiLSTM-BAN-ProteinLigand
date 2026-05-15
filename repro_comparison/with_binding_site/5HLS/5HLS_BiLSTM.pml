# PyMOL Script - 5HLS BiLSTM
# Attention frequency (n=5 ligands)
# RED>75%  ORANGE 50-75%  YELLOW 25-50%
# BLUE = direct contact <=5A  MAGENTA = RED+contact
# Usage: File -> Run Script (keep pdb/sdf in same folder)

reinitialize
load 5hls_protein.pdb, protein
load 5hls_ligand.sdf,  ligand
hide everything, all
show cartoon, protein
color gray70, protein
show sticks, ligand
color cyan, ligand

select attn_red, (protein and resi 42+43+44+45+46+66+67+68+69+70+71+72+73+74+75+76+77+78+79+80+81+82)
color red, attn_red
show sticks, attn_red

select attn_orange, (protein and resi 47)
color orange, attn_orange
show sticks, attn_orange

select attn_yellow, (protein and resi 55)
color yellow, attn_yellow
show sticks, attn_yellow

select binding_site, (protein and resi 81+82+83+85+87+88+92+94+97+136+139+140+145+146+149)
color blue, binding_site
show sticks, binding_site

select consensus, attn_red and binding_site
color magenta, consensus
show sticks, consensus

zoom
bg_color white

print "=== 5HLS BiLSTM ==="
print "RED    (22): [42, 43, 44, 45, 46, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82]"
print "Contact (15): [81, 82, 83, 85, 87, 88, 92, 94, 97, 136, 139, 140, 145, 146, 149]"
