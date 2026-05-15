# PyMOL Script - 5HLS BiLSTM + fpocket
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
load 5hls_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 5hls_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# MAGENTA: BiLSTM RED + fpocket consensus (6 residues)
select consensus, (protein and resi 44+45+46+79+81+82)
color magenta, consensus
show sticks, consensus

# RED: BiLSTM attention only (17 residues)
select attn_only, (protein and resi 42+43+55+66+67+68+69+70+71+72+73+74+75+76+77+78+80)
color red, attn_only
show sticks, attn_only

# BLUE: fpocket only (38 residues)
select fpocket_only, (protein and resi 47+48+50+83+84+85+86+87+88+91+92+94+97+98+103+104+105+106+108+109+112+113+128+131+132+136+137+139+140+141+142+145+146+147+148+149+151+154)
color blue, fpocket_only
show sticks, fpocket_only

# ORANGE: attention 50-75% (1 residues)
select attn_orange, (protein and resi 47)
color orange, attn_orange
show sticks, attn_orange

# YELLOW: attention 25-50% (1 residues)
select attn_yellow, (protein and resi 55)
color yellow, attn_yellow
show sticks, attn_yellow

zoom
bg_color white

print "=== 5HLS BiLSTM+fpocket ==="
print "MAGENTA consensus: 6 residues"
print "RED attn only    : 17 residues"
print "BLUE fpocket only: 38 residues"
