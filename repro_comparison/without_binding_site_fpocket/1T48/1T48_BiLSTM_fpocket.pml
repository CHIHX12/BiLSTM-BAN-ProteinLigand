# PyMOL Script - 1T48 BiLSTM + fpocket
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
load 1t48_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 1t48_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# MAGENTA: BiLSTM RED + fpocket consensus (15 residues)
select consensus, (protein and resi 1+2+3+145+157+158+159+168+169+170+176+178+190+192+193)
color magenta, consensus
show sticks, consensus

# RED: BiLSTM attention only (50 residues)
select attn_only, (protein and resi 4+5+6+7+8+117+118+119+127+128+129+130+131+132+133+134+135+136+137+138+139+140+141+142+143+144+146+147+148+149+151+160+161+162+163+164+165+166+167+171+172+173+174+175+177+191+201+203+204+205)
color red, attn_only
show sticks, attn_only

# BLUE: fpocket only (42 residues)
select fpocket_only, (protein and resi 73+74+75+78+79+80+152+180+185+186+187+188+189+196+197+200+232+235+236+240+241+242+244+252+256+276+277+278+279+280+281+282+288+289+291+292+293+294+295+296+297+298)
color blue, fpocket_only
show sticks, fpocket_only

# YELLOW: attention 25-50% (6 residues)
select attn_yellow, (protein and resi 8+127+128+129+176+193)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white

print "=== 1T48 BiLSTM+fpocket ==="
print "MAGENTA consensus: 15 residues"
print "RED attn only    : 50 residues"
print "BLUE fpocket only: 42 residues"
