# PyMOL Script - 1T48 BiLSTM
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

select attn_red, (protein and resi 1+2+3+4+5+6+7+117+118+119+130+131+132+133+134+135+136+137+138+139+140+141+142+143+144+145+146+147+148+149+151+157+158+159+160+161+162+163+164+165+166+167+168+169+170+171+172+173+174+175+177+178+190+191+192+201+203+204+205)
color red, attn_red
show sticks, attn_red

select attn_yellow, (protein and resi 8+127+128+129+176+193)
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

print "=== 1T48 BiLSTM ==="
print "RED    (59): [1, 2, 3, 4, 5, 6, 7, 117, 118, 119, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 151, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 177, 178, 190, 191, 192, 201, 203, 204, 205]"
print "Contact (22): [187, 188, 189, 192, 193, 195, 196, 197, 200, 232, 276, 277, 278, 279, 280, 281, 285, 286, 287, 288, 291, 292]"
