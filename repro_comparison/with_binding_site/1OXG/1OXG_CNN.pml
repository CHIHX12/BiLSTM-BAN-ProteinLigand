# PyMOL Script - 1OXG CNN
# Attention frequency (n=22 ligands)
# RED>75%  ORANGE 50-75%  YELLOW 25-50%
# BLUE = direct contact <=5A  MAGENTA = RED+contact
# Usage: File -> Run Script (keep pdb/sdf in same folder)

reinitialize
load 1oxg_protein.pdb, protein
load 1oxg_ligand.sdf,  ligand
hide everything, all
show cartoon, protein
color gray70, protein
show sticks, ligand
color cyan, ligand

select attn_red, (protein and resi 2+28+38+39+56+65+66+67+75+85+93+130+132+140+141+159+168+232+233)
color red, attn_red
show sticks, attn_red

select attn_orange, (protein and resi 150+160+234)
color orange, attn_orange
show sticks, attn_orange

select attn_yellow, (protein and resi 29+104+187)
color yellow, attn_yellow
show sticks, attn_yellow

select binding_site, (protein and resi 42+57+58+99+172+175+189+190+191+192+193+194+195+213+214+215+216+217+218+219+220+221+224+225+226+227+228)
color blue, binding_site
show sticks, binding_site

select consensus, attn_red and binding_site
color magenta, consensus
show sticks, consensus

zoom
bg_color white

print "=== 1OXG CNN ==="
print "RED    (19): [2, 28, 38, 39, 56, 65, 66, 67, 75, 85, 93, 130, 132, 140, 141, 159, 168, 232, 233]"
print "Contact (27): [42, 57, 58, 99, 172, 175, 189, 190, 191, 192, 193, 194, 195, 213, 214, 215, 216, 217, 218, 219, 220, 221, 224, 225, 226, 227, 228]"
