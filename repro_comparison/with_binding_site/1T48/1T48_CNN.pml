# PyMOL Script - 1T48 CNN
# Attention frequency (n=6 ligands)
# RED>75%  ORANGE 50-75%  YELLOW 25-50%
# BLUE = direct contact <=5A  MAGENTA = RED+contact
# Usage: File -> Run Script (keep pdb/sdf in same folder)

reinitialize
load 1t48_protein.pdb, protein
load 1t48_ligand.sdf,  ligand
hide everything, all
show cartoon, protein
color gray70, protein
show sticks, ligand
color cyan, ligand

select attn_red, (protein and resi 10+12+29+30+39+56+67+75+93+94+123+131+158+167+169+178+186+188+195+205+206+214+232+241+243+252+289+297)
color red, attn_red
show sticks, attn_red

select attn_orange, (protein and resi 132)
color orange, attn_orange
show sticks, attn_orange

select attn_yellow, (protein and resi 19+84+271)
color yellow, attn_yellow
show sticks, attn_yellow

select binding_site, (protein and resi 187+188+189+192+193+195+196+197+200+232+276+277+278+279+280+281+285+286+287+288+291+292)
color blue, binding_site
show sticks, binding_site

select consensus, attn_red and binding_site
color magenta, consensus
show sticks, consensus

zoom
bg_color white

print "=== 1T48 CNN ==="
print "RED    (28): [10, 12, 29, 30, 39, 56, 67, 75, 93, 94, 123, 131, 158, 167, 169, 178, 186, 188, 195, 205, 206, 214, 232, 241, 243, 252, 289, 297]"
print "Contact (22): [187, 188, 189, 192, 193, 195, 196, 197, 200, 232, 276, 277, 278, 279, 280, 281, 285, 286, 287, 288, 291, 292]"
