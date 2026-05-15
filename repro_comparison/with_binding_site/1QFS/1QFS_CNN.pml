# PyMOL Script - 1QFS CNN
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

select attn_red, (protein and resi 2+13+15+38+49+51+68+69+70+88+94+95+96+114+125+140+142+150+168+169+172+180+195+204+208+215+217+232+234+245+252+255+299+316+325+330+336+372+390+391+400+401+410+418+421+427+468+469+474+475+482+487+495+501+505+520+528+530+541+551+556+557+558+566+575+585+588+589+597+603+606+616+625+631+632+635+643+649+662+670+672+677+688+696+700+709)
color red, attn_red
show sticks, attn_red

select attn_orange, (protein and resi 30+84+319+320+328+367+430+706)
color orange, attn_orange
show sticks, attn_orange

select attn_yellow, (protein and resi 502+569+604)
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

print "=== 1QFS CNN ==="
print "RED    (86): [2, 13, 15, 38, 49, 51, 68, 69, 70, 88, 94, 95, 96, 114, 125, 140, 142, 150, 168, 169, 172, 180, 195, 204, 208, 215, 217, 232, 234, 245, 252, 255, 299, 316, 325, 330, 336, 372, 390, 391, 400, 401, 410, 418, 421, 427, 468, 469, 474, 475, 482, 487, 495, 501, 505, 520, 528, 530, 541, 551, 556, 557, 558, 566, 575, 585, 588, 589, 597, 603, 606, 616, 625, 631, 632, 635, 643, 649, 662, 670, 672, 677, 688, 696, 700, 709]"
print "Contact (21): [173, 235, 252, 254, 255, 473, 476, 478, 553, 554, 555, 556, 578, 580, 591, 594, 595, 599, 643, 644, 680]"
