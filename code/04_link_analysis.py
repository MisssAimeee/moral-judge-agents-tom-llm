#!/usr/bin/env python3
"""
04_link_analysis.py  --  Level 3: does representation predict behavior?

Across models, correlate:
  x = peak-layer intent-decoding accuracy            (from 02_probe.py)
  y = behavioral intent-reliance index               (from 03_behavioral.py)

In humans, rTPJ intent-encoding predicts forgiveness of accidents (Young & Saxe 2009).
The model analogue: do models that REPRESENT intent more separably also USE it more in
their judgments? A positive correlation = representation->behavior link. A null = the
"convergence without understanding" dissociation (still a publishable result).

Outputs: outputs/link/representation_vs_behavior.csv  + a scatter plot.
"""
import os, csv, glob, argparse

def peak_intent(probe_csv):
    best = None
    for r in csv.DictReader(open(probe_csv)):
        if r["target"]=="intent":
            acc=float(r["cv_acc"])
            if best is None or acc>best[1]: best=(int(r["layer"]),acc)
    return best  # (layer, acc)

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=os.path.join(here,"..","outputs","probe"))
    ap.add_argument("--behavior", default=os.path.join(here,"..","outputs","behavior","intent_reliance_summary.csv"))
    ap.add_argument("--out", default=os.path.join(here,"..","outputs","link"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rep = {}
    for p in glob.glob(os.path.join(a.probe,"*_probe.csv")):
        tag = os.path.basename(p).replace("_probe.csv","")
        L,acc = peak_intent(p); rep[tag]=(L,acc)
    beh = {r["model"].split("/")[-1]: float(r["intent_reliance_index"])
           for r in csv.DictReader(open(a.behavior))}

    rows=[]
    for tag,(L,acc) in rep.items():
        if tag in beh:
            rows.append([tag,L,round(acc,3),round(beh[tag],3)])
    outp=os.path.join(a.out,"representation_vs_behavior.csv")
    with open(outp,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["model","peak_intent_layer","peak_intent_acc","intent_reliance_index"]); w.writerows(rows)

    try:
        import numpy as np, matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if len(rows)>=3:
            x=np.array([r[2] for r in rows]); y=np.array([r[3] for r in rows])
            r=np.corrcoef(x,y)[0,1]
            plt.figure(figsize=(5,4))
            plt.scatter(x,y)
            for m,_,xx,yy in rows: plt.annotate(m,(xx,yy),fontsize=7)
            plt.xlabel("peak intent-decoding accuracy"); plt.ylabel("behavioral intent-reliance")
            plt.title(f"representation vs behavior  (r={r:.2f}, n={len(rows)})")
            plt.tight_layout(); plt.savefig(os.path.join(a.out,"rep_vs_behavior.png"),dpi=150)
            print(f"correlation r={r:.3f} over {len(rows)} models -> {outp}")
    except Exception as e:
        print("plot skipped:", e)
