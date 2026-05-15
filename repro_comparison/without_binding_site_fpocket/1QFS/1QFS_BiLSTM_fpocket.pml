# PyMOL Script - 1QFS BiLSTM + fpocket
#
# Color scheme:
#   MAGENTA : Consensus = BiLSTM RED and fpocket  (strongest signal)
#   RED     : BiLSTM attention only (>75%, not in fpocket)
#   BLUE    : fpocket only (not in BiLSTM RED)
#   ORANGE  : BiLSTM attention 50-75%
#   YELLOW  : BiLSTM attention 25-50%
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

# MAGENTA: BiLSTM RED + fpocket consensus (5 residues)
select consensus, (protein and resi 174+196+200+250+272)
color magenta, consensus
show sticks, consensus

# RED: BiLSTM attention only (89 residues)
select attn_only, (protein and resi 1+2+3+4+5+6+7+8+103+104+105+106+107+108+113+114+115+116+118+119+120+127+129+131+132+133+134+135+136+137+138+139+140+141+142+143+144+145+146+147+148+149+150+151+152+153+154+155+156+159+160+168+192+193+194+195+197+198+199+201+202+203+215+216+220+241+248+249+251+270+369+371+372+386+387+388+391+393+394+395+397+398+399+400+401+402+403+404+405)
color red, attn_only
show sticks, attn_only

# BLUE: fpocket only (54 residues)
select fpocket_only, (protein and resi 10+12+30+33+34+35+36+38+39+40+43+47+173+175+179+180+181+183+188+208+212+214+217+221+223+235+237+244+252+254+255+266+271+273+274+296+299+301+315+345+346+347+366+367+591+594+595+639+643+646+650+674+675+676)
color blue, fpocket_only
show sticks, fpocket_only

# ORANGE: attention 50-75% (3 residues)
select attn_orange, (protein and resi 157+158+175)
color orange, attn_orange
show sticks, attn_orange

# YELLOW: attention 25-50% (5 residues)
select attn_yellow, (protein and resi 103+118+168+241+391)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white

print "=== 1QFS BiLSTM+fpocket ==="
print "MAGENTA consensus: 5 residues"
print "RED attn only    : 89 residues"
print "BLUE fpocket only: 54 residues"
