"""Plot full-precision Rust outputs and check the independent NumPy oracle."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
def load(path):
    return json.loads((ROOT / path).read_text())

def main():
    cases = ['prior', 'AD', 'ADX']
    runs = [load(f'artifacts/{c}/run.json') for c in cases]
    a, b = np.array([[1, 2], [3, 4]]), np.array([[2, 0], [1, 2]])
    ab = np.array([[sum(a[i,j]*b[j,k] for j in range(2)) for k in range(2)] for i in range(2)])
    checks=[]
    for run in runs:
        c=run['checks']
        errors=[float(np.max(np.abs(ab-np.array(c['ab_row_major']).reshape(2,2)))), abs(float(np.trace(ab))-c['trace']), c['ising_absolute_error']]
        assert max(errors)<1e-12
        checks.append(errors)
    print('NumPy AB:',ab.tolist(),'trace:',np.trace(ab),'absolute errors:',checks)
    fig, ax = plt.subplots(figsize=(8,4))
    for i, (case, run) in enumerate(zip(cases, runs)):
        x=np.arange(3)+(i-1)*0.25
        ax.bar(x,[run['node_marginals'][v][0] for v in [1,3,4]],width=.23,label=case)
        ax.scatter(x,[run['enumeration_marginals'][v][0] for v in [1,3,4]],marker='x',color='black',s=45)
    ax.set(xticks=range(3),xticklabels=['Tuberculosis','Lung cancer','Bronchitis'],ylabel='Probability of state 0',ylim=(0,1),title='ASIA: bars = contraction; crosses = enumeration')
    ax.legend();fig.tight_layout();fig.savefig(ROOT/'evidence/posteriors.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(9,8))
    for i,(case,run) in enumerate(zip(cases,runs)):
        joint=np.array(run['joint_TL']).reshape(2,2)
        product=np.outer(run['node_marginals'][1],run['node_marginals'][3])
        for j,(title,arr) in enumerate([('Joint',joint),('Product of marginals',product),('Dependence: joint - product',joint-product)]):
            ax=axes[i,j];ax.imshow(arr,cmap='RdBu_r' if j==2 else 'Blues',vmin=-.5 if j==2 else 0,vmax=.5 if j==2 else 1)
            for (r,c),v in np.ndenumerate(arr):ax.text(c,r,f'{v:.6f}',ha='center',va='center',color='black',bbox=dict(facecolor='white',alpha=.7,edgecolor='none'))
            ax.set(xticks=[0,1],yticks=[0,1],xlabel='L state',ylabel=f'{case}: T state')
            if i==0:ax.set_title(title,fontsize=10)
    fig.tight_layout();fig.savefig(ROOT/'evidence/joint.png',dpi=170);plt.close(fig)
    (ROOT/'evidence/asia.json').write_text(json.dumps(dict(zip(cases,runs)),indent=2)+'\n')
    (ROOT/'evidence/numpy_checks.json').write_text(json.dumps({'ab':ab.tolist(),'trace':int(np.trace(ab)),'absolute_errors':checks},indent=2)+'\n')
if __name__=='__main__':main()
