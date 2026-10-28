"""Shared paths, reproducible output and statistical checks."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MPLCONFIGDIR', str(__import__('pathlib').Path(__file__).resolve().parents[1]/'artifacts/mpl'))
from pathlib import Path
import json
import numpy as np
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; ART=ROOT/'artifacts'; EVI=ROOT/'evidence'
for p in (DATA,ART,EVI): p.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi':130,'axes.spines.top':False,'axes.spines.right':False})
def save(name,obj):
    (EVI/f'{name}.json').write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda v: v.tolist() if hasattr(v,'tolist') else float(v)))
def fig(name):
    plt.tight_layout(); plt.savefig(EVI/f'{name}.png'); plt.close()
def chi_stat(obs,p):
    expected=np.sum(obs)*p
    good=expected>=5
    o=list(obs[good]); e=list(expected[good])
    if expected[~good].sum()>0:
        if expected[~good].sum()<5:
            j=int(np.argmax(e)); o[j]+=obs[~good].sum(); e[j]+=expected[~good].sum()
        else: o.append(obs[~good].sum()); e.append(expected[~good].sum())
    stat=np.sum((np.array(o)-e)**2/np.array(e)); df=len(e)-1
    return float(stat),df
def chi_test(obs,p):
    s,d=chi_stat(obs,p); return {'statistic':s,'df':d,'p':float(chi2.sf(s,d))}
def corpus():
    return np.load(ART/'tokens.npz'),json.loads((EVI/'corpus.json').read_text())
