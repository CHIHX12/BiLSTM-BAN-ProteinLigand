# PyMOL Script - 5HLS CNN
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

select attn_red, (protein and resi 52+60+61+69+70+80+81+82+83+84+85+86+87+97+107+116+126+135+144+145+146+147+148+149+150+151+152)
color red, attn_red
show sticks, attn_red

select binding_site, (protein and resi 81+82+83+85+87+88+92+94+97+136+139+140+145+146+149)
color blue, binding_site
show sticks, binding_site

select consensus, attn_red and binding_site
color magenta, consensus
show sticks, consensus

zoom
bg_color white

print "=== 5HLS CNN ==="
print "RED    (27): [52, 60, 61, 69, 70, 80, 81, 82, 83, 84, 85, 86, 87, 97, 107, 116, 126, 135, 144, 145, 146, 147, 148, 149, 150, 151, 152]"
print "Contact (15): [81, 82, 83, 85, 87, 88, 92, 94, 97, 136, 139, 140, 145, 146, 149]"
