# PyMOL Script - 5HLS CNN + fpocket
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
load 5hls_protein.pdb, protein
hide everything, all
show cartoon, protein
color gray70, protein

load 5hls_ligand.sdf, ligand
show sticks, ligand
color cyan, ligand

# MAGENTA: CNN RED + fpocket consensus (14 residues)
select consensus, (protein and resi 81+82+83+84+85+86+87+97+145+146+147+148+149+151)
color magenta, consensus
show sticks, consensus

# RED: CNN attention only (13 residues)
select attn_only, (protein and resi 52+60+61+69+70+80+107+116+126+135+144+150+152)
color red, attn_only
show sticks, attn_only

# BLUE: fpocket only (30 residues)
select fpocket_only, (protein and resi 44+45+46+47+48+50+79+88+91+92+94+98+103+104+105+106+108+109+112+113+128+131+132+136+137+139+140+141+142+154)
color blue, fpocket_only
show sticks, fpocket_only

zoom
bg_color white

print "=== 5HLS CNN+fpocket ==="
print "MAGENTA consensus: 14 residues"
print "RED attn only    : 13 residues"
print "BLUE fpocket only: 30 residues"
