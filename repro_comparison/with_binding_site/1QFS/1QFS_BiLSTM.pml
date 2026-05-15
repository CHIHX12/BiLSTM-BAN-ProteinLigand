# PyMOL Script - 1QFS BiLSTM
# Attention frequency (n=17 ligands)
# RED>75%  ORANGE 50-75%  YELLOW 25-50%
# BLUE = direct contact <=5A  MAGENTA = RED+contact
# Usage: File -> Run Script (keep pdb/sdf in same folder)

reinitialize
load 1qfs_protein.pdb, protein
load 1qfs_ligand.sdf,  ligand
hide everything, all
show cartoon, protein
color gray70, protein
show sticks, ligand
color cyan, ligand

select attn_red, (protein and resi 1+2+3+4+5+6+7+8+104+105+106+107+108+113+114+115+116+119+120+127+129+131+132+133+134+135+136+137+138+139+140+141+142+143+144+145+146+147+148+149+150+151+152+153+154+155+156+159+160+174+192+193+194+195+196+197+198+199+200+201+202+203+215+216+220+248+249+250+251+270+272+369+371+372+386+387+388+393+394+395+397+398+399+400+401+402+403+404+405)
color red, attn_red
show sticks, attn_red

select attn_orange, (protein and resi 157+158+175)
color orange, attn_orange
show sticks, attn_orange

select attn_yellow, (protein and resi 103+118+168+241+391)
color yellow, attn_yellow
show sticks, attn_yellow

select binding_site, (protein and resi 173+235+252+254+255+473+476+478+553+554+555+556+578+580+591+594+595+599+643+644+680)
color blue, binding_site
show sticks, binding_site

select consensus, attn_red and binding_site
color magenta, consensus
show sticks, consensus

zoom
bg_color white

print "=== 1QFS BiLSTM ==="
print "RED    (89): [1, 2, 3, 4, 5, 6, 7, 8, 104, 105, 106, 107, 108, 113, 114, 115, 116, 119, 120, 127, 129, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 159, 160, 174, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 215, 216, 220, 248, 249, 250, 251, 270, 272, 369, 371, 372, 386, 387, 388, 393, 394, 395, 397, 398, 399, 400, 401, 402, 403, 404, 405]"
print "Contact (21): [173, 235, 252, 254, 255, 473, 476, 478, 553, 554, 555, 556, 578, 580, 591, 594, 595, 599, 643, 644, 680]"
