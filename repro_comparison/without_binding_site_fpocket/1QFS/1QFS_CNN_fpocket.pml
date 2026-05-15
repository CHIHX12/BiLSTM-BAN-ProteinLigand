# PyMOL Script - 1QFS CNN + fpocket
#
# Color scheme:
#   MAGENTA : Consensus = CNN RED and fpocket  (strongest signal)
#   RED     : CNN attention only (>75%, not in fpocket)
#   BLUE    : fpocket only (not in CNN RED)
#   ORANGE  : CNN attention 50-75%
#   YELLOW  : CNN attention 25-50%
#   CYAN    : Ligand
#   GRAY    : Other residues
#
# Usage: File -> Run Script (keep pdb/sdf in same folder)

reinitialize
load 1qfs_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 1qfs_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# MAGENTA: CNN RED + fpocket consensus (8 residues)
select consensus, (protein and resi 38+180+208+217+252+255+299+643)
color magenta, consensus
show sticks, consensus

# RED: CNN attention only (78 residues)
select attn_only, (protein and resi 2+13+15+49+51+68+69+70+88+94+95+96+114+125+140+142+150+168+169+172+195+204+215+232+234+245+316+325+330+336+372+390+391+400+401+410+418+421+427+468+469+474+475+482+487+495+501+505+520+528+530+541+551+556+557+558+566+575+585+588+589+597+603+606+616+625+631+632+635+649+662+670+672+677+688+696+700+709)
color red, attn_only
show sticks, attn_only

# BLUE: fpocket only (51 residues)
select fpocket_only, (protein and resi 10+12+30+33+34+35+36+39+40+43+47+173+174+175+179+181+183+188+196+200+212+214+221+223+235+237+244+250+254+266+271+272+273+274+296+301+315+345+346+347+366+367+591+594+595+639+646+650+674+675+676)
color blue, fpocket_only
show sticks, fpocket_only

# ORANGE: attention 50-75% (8 residues)
select attn_orange, (protein and resi 30+84+319+320+328+367+430+706)
color orange, attn_orange
show sticks, attn_orange

# YELLOW: attention 25-50% (3 residues)
select attn_yellow, (protein and resi 502+569+604)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white

print "=== 1QFS CNN+fpocket ==="
print "MAGENTA consensus: 8 residues"
print "RED attn only    : 78 residues"
print "BLUE fpocket only: 51 residues"
