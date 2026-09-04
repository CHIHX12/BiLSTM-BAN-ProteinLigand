#!/usr/bin/env python3
"""
teiban_web.py -- a tiny, dependency-free web UI for TEIBAN screening.

Runs on the HOST (Slurm login node) with the system python3 (STANDARD LIBRARY
ONLY -- no torch/dgl/rdkit/flask). It lets a user, from a browser:

  * browse folders under $HOME and pick a SMILES file,
  * paste ONE protein sequence,
  * submit a "1 protein x many SMILES" screen to the GPU cluster (multi-GPU via
    a Slurm array), and
  * watch a progress bar until the merged results CSV is ready.

The heavy lifting (SMILES cleaning + model inference) happens INSIDE teiban.sif
on the GPU nodes, driven by submit_teiban.sh. This server only orchestrates:
it shells out to `singularity exec teiban.sif ...` and `sbatch`, which is why it
must run on the login node (where those commands live), not inside the container.

LAUNCH (on the login node):
    singularity exec teiban.sif cat /opt/teiban/teiban_web.py > teiban_web.py
    python3 teiban_web.py                      # then open the printed URL

OPTIONS:
    --host 127.0.0.1   bind address (default: localhost only, safest)
    --port 8700        port
    --root ~           highest folder the browser may read/write (default: $HOME)
    --sif  PATH        path to teiban.sif (default: auto-detect near this script)
"""
import argparse
import glob
import html
import json
import os
import re
import shutil
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

CFG = {"root": os.path.expanduser("~"), "sif": None, "partition": "all"}
SMILES_EXTS = (".smi", ".txt", ".csv", ".tsv", ".ism", ".smiles")


# ---------------------------------------------------------------------------
# Helpers (all confined under CFG["root"])
# ---------------------------------------------------------------------------
def safe_path(path):
    """Resolve `path` and confine it under root. Returns an abspath or None."""
    root = os.path.realpath(CFG["root"])
    if not path:
        return root
    p = path if os.path.isabs(path) else os.path.join(root, path)
    p = os.path.realpath(p)
    return p if p == root or p.startswith(root + os.sep) else None


def find_sif():
    if CFG["sif"] and os.path.isfile(CFG["sif"]):
        return CFG["sif"]
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.getcwd(), here):
        for name in ("teiban-internal.sif", "teiban.sif"):
            cand = os.path.join(d, name)
            if os.path.isfile(cand):
                return cand
    return None


def list_dir(path):
    p = safe_path(path)
    if not p or not os.path.isdir(p):
        return {"error": "folder not accessible"}
    root = os.path.realpath(CFG["root"])
    up = None if p == root else os.path.dirname(p)
    dirs, files = [], []
    try:
        names = sorted(os.listdir(p), key=str.lower)
    except OSError as e:
        return {"error": str(e)}
    for name in names:
        if name.startswith("."):
            continue
        fp = os.path.join(p, name)
        try:
            if os.path.isdir(fp):
                dirs.append({"name": name, "path": fp})
            elif os.path.isfile(fp):
                ext = os.path.splitext(name)[1].lower()
                files.append({"name": name, "path": fp, "size": os.path.getsize(fp),
                              "smiles": ext in SMILES_EXTS})
        except OSError:
            continue
    return {"path": p, "up": up, "dirs": dirs, "files": files}


def head_file(path, n=15):
    p = safe_path(path)
    if not p or not os.path.isfile(p):
        return {"error": "file not accessible"}
    lines, count = [], 0
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            for i, ln in enumerate(f):
                count += 1
                if i < n:
                    lines.append(ln.rstrip("\n")[:200])
    except OSError as e:
        return {"error": str(e)}
    return {"path": p, "lines": lines, "count": count}


def submit(data):
    sif = find_sif()
    if not sif:
        return {"ok": False, "error": "teiban.sif not found -- start me next to it or pass --sif"}
    protein = (data.get("protein") or "").strip()
    if not protein or len(re.sub(r"[^A-Za-z]", "", protein)) < 20:
        return {"ok": False, "error": "please paste a protein sequence (>= 20 residues)"}
    smiles_path = safe_path(data.get("smiles_path"))
    if not smiles_path or not os.path.isfile(smiles_path):
        return {"ok": False, "error": "pick a SMILES file from the browser"}
    workdir = safe_path(data.get("workdir")) or os.path.realpath(CFG["root"])
    if not os.path.isdir(workdir):
        return {"ok": False, "error": "the working folder is not valid"}
    gpus = str(data.get("gpus") or "1")
    gpus = int(gpus) if gpus.isdigit() and int(gpus) >= 1 else 1
    partition = (data.get("partition") or CFG["partition"]).strip() or CFG["partition"]
    model = (data.get("model") or "BiLSTM").strip()
    if model not in ("BiLSTM", "CNN", "both"):
        model = "BiLSTM"
    pid = (data.get("protein_id") or "target").strip() or "target"
    out_name = (data.get("out_name") or "screen_pred.csv").strip() or "screen_pred.csv"
    if not out_name.lower().endswith(".csv"):
        out_name += ".csv"

    tag = time.strftime("%Y%m%d_%H%M%S")
    prot_file = os.path.join(workdir, f"screen_{tag}_protein.txt")
    pairs_csv = os.path.join(workdir, f"screen_{tag}_pairs.csv")
    out_csv = os.path.join(workdir, out_name)
    with open(prot_file, "w", encoding="utf-8") as f:
        f.write(protein)

    # 1) Build a clean pairs CSV INSIDE the sif (rdkit lives there, not on host).
    build = subprocess.run(
        ["singularity", "exec", sif, "python3", "/opt/teiban/predict_simple.py",
         "--screen-csv", "--protein-file", prot_file, "--protein-id", pid,
         "--drug-file", smiles_path, "--output", pairs_csv],
        capture_output=True, text=True)
    if build.returncode != 0 or not os.path.isfile(pairs_csv):
        return {"ok": False, "error": "could not build pairs CSV: "
                + (build.stdout + build.stderr).strip()[-400:]}
    npairs = max(0, sum(1 for _ in open(pairs_csv, encoding="utf-8", errors="replace")) - 1)
    if npairs == 0:
        return {"ok": False, "error": "no valid drug-protein pairs after cleaning the SMILES"}
    chunks = max(1, min(gpus, npairs))

    # 2) Submit to Slurm via the bundled helper (extracted from the sif).
    submit_sh = os.path.join(workdir, "submit_teiban.sh")
    subprocess.run(f"singularity exec {sif} cat /opt/teiban/submit_teiban.sh > "
                   f"{submit_sh!r}".replace("'", '"'), shell=True)
    run = subprocess.run(
        ["bash", submit_sh, "--input", pairs_csv, "--output", out_csv,
         "--partition", partition, "--chunks", str(chunks), "--model", model, "--sif", sif],
        capture_output=True, text=True, cwd=workdir)
    out = run.stdout + run.stderr
    mjob = re.search(r"array job:\s*(\d+)", out) or re.search(r"Submitted batch job (\d+)", out)
    mdir = re.search(r"chunks -> (\S+)", out)
    job = mjob.group(1) if mjob else ""
    chunks_dir = mdir.group(1) if mdir else ""
    if not job and not chunks_dir:
        return {"ok": False, "error": "Slurm submit failed: " + out.strip()[-400:]}
    return {"ok": True, "job": job, "chunks_dir": chunks_dir, "total": chunks,
            "pairs": npairs, "out": out_csv, "partition": partition, "gpus": chunks,
            "log": out.strip()[-600:]}


def progress(qs):
    cd = (qs.get("dir", [""])[0] or "").strip()
    out = (qs.get("out", [""])[0] or "").strip()
    job = (qs.get("job", [""])[0] or "").strip()
    total = int(qs.get("total", ["0"])[0] or 0)
    res = {"total": total, "done": 0, "running": 0, "state": "", "merged": False, "rows": None}
    if cd and os.path.isdir(cd):
        parts = glob.glob(os.path.join(cd, "part_*.csv"))
        preds = glob.glob(os.path.join(cd, "pred_*.csv"))
        res["total"] = len(parts) or total
        res["done"] = len(preds)
    if out and os.path.isfile(out):
        res["merged"] = True
        res["done"] = res["total"] or res["done"]
        try:
            res["rows"] = max(0, sum(1 for _ in open(out, encoding="utf-8", errors="replace")) - 1)
        except OSError:
            pass
    if shutil.which("squeue") and job:
        r = subprocess.run(["squeue", "-j", job, "-h", "-o", "%T"], capture_output=True, text=True)
        states = [s for s in r.stdout.split() if s]
        res["running"] = sum(1 for s in states if s == "RUNNING")
        res["state"] = ",".join(sorted(set(states))) or ("done" if res["merged"] else "queued/finished")
    elif res["merged"]:
        res["state"] = "done"
    return res


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass  # quiet

    def do_GET(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if u.path == "/api/ls":
            return self._send(200, list_dir(qs.get("path", [""])[0]))
        if u.path == "/api/head":
            return self._send(200, head_file(qs.get("path", [""])[0], int(qs.get("n", ["15"])[0])))
        if u.path == "/api/progress":
            return self._send(200, progress(qs))
        if u.path == "/api/download":
            p = safe_path(qs.get("path", [""])[0])
            if not p or not os.path.isfile(p):
                return self._send(404, {"error": "not found"})
            with open(p, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Disposition",
                             f'attachment; filename="{os.path.basename(p)}"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)
        return self._send(404, {"error": "no such endpoint"})

    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self._send(400, {"ok": False, "error": "bad JSON"})
        if u.path == "/api/submit":
            try:
                return self._send(200, submit(data))
            except Exception as e:
                return self._send(200, {"ok": False, "error": f"server error: {e}"})
        return self._send(404, {"ok": False, "error": "no such endpoint"})


# ---------------------------------------------------------------------------
# The single-page UI (inline; no external resources)
# ---------------------------------------------------------------------------
PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TEIBAN screening</title>
<style>
:root{--bg:#0e1116;--panel:#161b22;--panel2:#1c232d;--line:#2a3340;--txt:#e6edf3;
--muted:#8b98a5;--accent:#3fb950;--accent2:#2f81f7;--warn:#e3b341;--radius:10px}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
background:var(--bg);color:var(--txt)}
header{padding:16px 22px;border-bottom:1px solid var(--line);display:flex;
align-items:baseline;gap:14px;background:linear-gradient(180deg,#11161d,#0e1116)}
header h1{margin:0;font-size:18px;letter-spacing:.5px}
header .tag{color:var(--accent);font-weight:700}
header small{color:var(--muted)}
.wrap{display:grid;grid-template-columns:minmax(320px,1fr) minmax(340px,1fr);
gap:18px;padding:18px;max-width:1200px;margin:0 auto}
.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);
overflow:hidden}
.card h2{margin:0;padding:12px 16px;font-size:13px;text-transform:uppercase;
letter-spacing:.8px;color:var(--muted);border-bottom:1px solid var(--line);background:var(--panel2)}
.card .body{padding:14px 16px}
.crumb{font-size:12px;color:var(--muted);word-break:break-all;margin-bottom:10px}
.list{max-height:340px;overflow:auto;border:1px solid var(--line);border-radius:8px}
.row{display:flex;align-items:center;gap:8px;padding:7px 11px;cursor:pointer;
border-bottom:1px solid #202834}
.row:last-child{border-bottom:0}
.row:hover{background:#20283400}
.row:hover{background:#1e2733}
.row .ic{width:16px;text-align:center;opacity:.85}
.row .nm{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.row .sz{color:var(--muted);font-size:11px}
.row.sel{background:#14351f;outline:1px solid var(--accent)}
.row.smiles .nm{color:#cfe8d4}
label{display:block;font-size:12px;color:var(--muted);margin:12px 0 4px}
input,select,textarea{width:100%;background:#0d1117;border:1px solid var(--line);
color:var(--txt);border-radius:8px;padding:9px 10px;font:inherit}
textarea{resize:vertical;min-height:84px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.pick{background:var(--panel2);border:1px dashed var(--line);border-radius:8px;
padding:10px;font-size:12px;color:var(--muted);margin-top:4px}
.pick b{color:var(--txt)}
button{margin-top:16px;width:100%;padding:11px;border:0;border-radius:8px;
background:var(--accent);color:#04140a;font-weight:700;font-size:14px;cursor:pointer}
button:disabled{opacity:.5;cursor:not-allowed}
button.ghost{background:var(--panel2);color:var(--txt);border:1px solid var(--line);font-weight:600}
.prog{margin-top:14px;display:none}
.bar{height:14px;background:#0d1117;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.bar>i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--accent2),var(--accent));
transition:width .4s}
.pstat{font-size:12px;color:var(--muted);margin-top:8px;display:flex;justify-content:space-between}
.note{font-size:12px;color:var(--warn);margin-top:8px;white-space:pre-wrap}
.ok{color:var(--accent)}
a.dl{color:var(--accent2);text-decoration:none}
</style></head><body>
<header><h1><span class="tag">TEIBAN</span> screening</h1>
<small>1 protein &times; many SMILES &rarr; Slurm (multi-GPU) &rarr; results CSV</small></header>
<div class="wrap">
  <div class="card">
    <h2>1 &mdash; pick a SMILES file &amp; working folder</h2>
    <div class="body">
      <div class="crumb" id="crumb">/</div>
      <div class="list" id="list"></div>
      <div class="pick">SMILES file: <b id="pkFile">(none)</b> <span id="pkCount"></span><br>
        output folder: <b id="pkDir">&mdash;</b></div>
    </div>
  </div>
  <div class="card">
    <h2>2 &mdash; protein &amp; run</h2>
    <div class="body">
      <label>Protein sequence (one target)</label>
      <textarea id="prot" placeholder="MENFQK... (paste the amino-acid sequence)"></textarea>
      <div class="grid3">
        <div><label>Protein id</label><input id="pid" value="target"></div>
        <div><label>GPUs</label><input id="gpus" type="number" min="1" max="16" value="4"></div>
        <div><label>Partition</label><select id="part"><option>all</option><option>intel</option></select></div>
      </div>
      <div class="grid3">
        <div><label>Model</label><select id="model"><option>BiLSTM</option><option>CNN</option><option>both</option></select></div>
        <div style="grid-column:span 2"><label>Output CSV name</label><input id="out" value="screen_pred.csv"></div>
      </div>
      <button id="go" disabled>Submit screen to cluster</button>
      <div class="prog" id="prog">
        <div class="bar"><i id="fill"></i></div>
        <div class="pstat"><span id="pmsg">submitting...</span><span id="ppct"></span></div>
        <div class="note" id="pnote"></div>
      </div>
    </div>
  </div>
</div>
<script>
const $=s=>document.querySelector(s);
let cwd=null, sel=null, run=null, timer=null;

async function ls(path){
  const r=await fetch('/api/ls?path='+encodeURIComponent(path||''));
  const d=await r.json();
  if(d.error){$('#list').innerHTML='<div class="row">'+d.error+'</div>';return;}
  cwd=d.path; $('#crumb').textContent=d.path; $('#pkDir').textContent=d.path;
  let h='';
  if(d.up!==null) h+=row('..',d.up,'📁','dir');
  d.dirs.forEach(x=>h+=row(x.name,x.path,'📁','dir'));
  d.files.forEach(x=>h+=row(x.name,x.path,x.smiles?'🧬':'📄',x.smiles?'file smiles':'file',x.size));
  $('#list').innerHTML=h;
  document.querySelectorAll('#list .row').forEach(el=>{
    el.onclick=()=>{ const p=el.dataset.p,k=el.dataset.k;
      if(k.startsWith('dir')) ls(p);
      else{ sel=p; $('#pkFile').textContent=p.split('/').pop();
        document.querySelectorAll('#list .row').forEach(r=>r.classList.remove('sel'));
        el.classList.add('sel'); head(p); check(); } };
  });
}
function row(name,path,ic,kind,size){
  const sz=size!=null?'<span class="sz">'+fmt(size)+'</span>':'';
  return '<div class="row '+kind+'" data-p="'+enc(path)+'" data-k="'+kind+'">'
    +'<span class="ic">'+ic+'</span><span class="nm">'+esc(name)+'</span>'+sz+'</div>';
}
function fmt(n){return n<1024?n+' B':n<1048576?(n/1024).toFixed(0)+' KB':(n/1048576).toFixed(1)+' MB';}
function esc(s){return s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function enc(s){return s.replace(/"/g,'&quot;');}
async function head(p){
  const r=await fetch('/api/head?path='+encodeURIComponent(p)+'&n=1');
  const d=await r.json();
  $('#pkCount').textContent=d.count!=null?'('+d.count+' lines)':'';
}
function check(){ $('#go').disabled=!(sel && $('#prot').value.trim().length>=20); }
$('#prot').addEventListener('input',check);

$('#go').onclick=async()=>{
  $('#go').disabled=true; $('#prog').style.display='block';
  $('#fill').style.width='0'; $('#pmsg').textContent='building pairs & submitting...';
  $('#ppct').textContent=''; $('#pnote').textContent='';
  const body={protein:$('#prot').value,protein_id:$('#pid').value,smiles_path:sel,
    workdir:cwd,gpus:$('#gpus').value,partition:$('#part').value,
    model:$('#model').value,out_name:$('#out').value};
  const r=await fetch('/api/submit',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)});
  const d=await r.json();
  if(!d.ok){ $('#pmsg').textContent='failed'; $('#pnote').textContent=d.error||'error';
    $('#go').disabled=false; return; }
  run=d; $('#pmsg').textContent='job '+(d.job||'?')+' submitted &middot; '+d.pairs+' pairs on '+d.gpus+' GPU(s)';
  poll();
  timer=setInterval(poll,3000);
};
async function poll(){
  if(!run) return;
  const q='dir='+encodeURIComponent(run.chunks_dir||'')+'&out='+encodeURIComponent(run.out)
    +'&job='+encodeURIComponent(run.job||'')+'&total='+(run.total||1);
  const d=await(await fetch('/api/progress?'+q)).json();
  const tot=d.total||run.total||1, done=Math.min(d.done||0,tot);
  const pct=Math.round(100*done/tot);
  $('#fill').style.width=(d.merged?100:pct)+'%'; $('#ppct').textContent=(d.merged?100:pct)+'%';
  let s='state: '+(d.state||'...')+'  ('+done+'/'+tot+' chunks';
  if(d.running) s+=', '+d.running+' running'; s+=')';
  $('#pmsg').innerHTML=s;
  if(d.merged){ clearInterval(timer);
    $('#pmsg').innerHTML='<span class="ok">done &middot; '+(d.rows!=null?d.rows+' predictions':'')+'</span>';
    $('#pnote').innerHTML='result: <a class="dl" href="/api/download?path='
      +encodeURIComponent(run.out)+'">'+esc(run.out.split('/').pop())+'</a>';
    $('#go').disabled=false;
  }
}
ls('');
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description="TEIBAN screening web UI (stdlib only).")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost)")
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--root", default=os.path.expanduser("~"),
                    help="highest folder the browser may read/write (default: $HOME)")
    ap.add_argument("--sif", help="path to teiban.sif (default: auto-detect near this script)")
    ap.add_argument("--partition", default="all", help="default Slurm partition (all/intel)")
    a = ap.parse_args()
    CFG["root"] = os.path.realpath(os.path.expanduser(a.root))
    CFG["sif"] = os.path.abspath(os.path.expanduser(a.sif)) if a.sif else None
    CFG["partition"] = a.partition
    sif = find_sif()
    if not shutil.which("sbatch"):
        print("  WARNING: 'sbatch' not found -- run this on the Slurm login node, not inside the container.")
    print("=" * 62)
    print("  TEIBAN screening web UI")
    print(f"  root folder : {CFG['root']}")
    print(f"  teiban.sif  : {sif or 'NOT FOUND (pass --sif)'}")
    print(f"  open in your browser:  http://{a.host}:{a.port}")
    if a.host == "127.0.0.1":
        print(f"  (localhost only -- from your laptop use:  ssh -L {a.port}:localhost:{a.port} <this-host>)")
    print("  Ctrl-C to stop.")
    print("=" * 62)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
