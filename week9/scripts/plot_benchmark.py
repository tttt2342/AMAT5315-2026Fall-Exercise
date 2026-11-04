"""Compare exact scales, selected marginals, BP traces, and plan costs."""
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_orders import recount
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def published():
    tokens=(ROOT/'promedus/Promedus_14.uai.MAR').read_text().split();assert tokens.pop(0)=='MAR'
    n=int(tokens.pop(0));values=[];i=0
    for _ in range(n):
        d=int(tokens[i]);values.append(list(map(float,tokens[i+1:i+d+1])));i+=d+1
    assert i==len(tokens)
    pr=(ROOT/'promedus/Promedus_14.uai.PR').read_text().split();assert pr[0]=='PR'
    return float(pr[1]),np.array(values)
def main():
    ids=[59,102,139];pr,mar=published();bp=read('artifacts/promedus-bp/run.json')
    runs={m:read(f'runs/promedus-{m}/run.json') for m in ['greedy','treesa']}
    costs={m:read(f'runs/promedus-{m}/complexity.json') for m in runs}
    checks={}
    for method,run in runs.items():
        prefix=f'runs/promedus-{method}/'
        recount(read(prefix+'order.json'),costs[method])
        for q in ids:recount(read(prefix+f'query-{q}-order.json'),read(prefix+f'query-{q}-complexity.json'))
        pr_error=abs(run['log10_partition']-pr)
        mar_error=max(abs(run['selected_marginals'][str(q)][s]-mar[q,s]) for q in ids for s in range(2))
        assert pr_error<1e-4 and mar_error<1e-5
        checks[method]=dict(log10_partition=run['log10_partition'],published_PR_error=pr_error,published_MAR_error=mar_error,selected_marginals=run['selected_marginals'],work=costs[method]['work'],largest_tensor=costs[method]['largest_tensor'],largest_generated_tensor=costs[method]['largest_generated_tensor'],planning_seconds=run['planning_seconds'],scalar_contraction_seconds=run['scalar_contraction_seconds'],runtime_seconds=run['runtime_seconds'],query_costs=run['query_costs'])
    diff=abs(runs['greedy']['log10_partition']-runs['treesa']['log10_partition'])
    qdiff=max(abs(runs['greedy']['selected_marginals'][str(q)][s]-runs['treesa']['selected_marginals'][str(q)][s]) for q in ids for s in range(2))
    assert max(diff,qdiff)<1e-10
    h=bp['history'];sweeps=[x['sweep'] for x in h];residual=[x['residual'] for x in h]
    errors=[float(np.max(np.abs(np.array(x['node_beliefs'])[ids]-mar[ids]))) for x in h]
    bp_selected=np.array(bp['node_marginals'])[ids]
    checks['bp']={k:bp[k] for k in ['sweeps','residual','converged','cumulative_update_work','runtime_seconds']}
    checks['bp'].update(selected_marginals=dict(zip(map(str,ids),bp_selected.tolist())),published_MAR_error=errors[-1])
    checks['exact_plan_max_difference']=max(diff,qdiff)
    checks['work_speedup']=costs['greedy']['work']/costs['treesa']['work']
    checks['largest_tensor_ratio']=costs['greedy']['largest_tensor']/costs['treesa']['largest_tensor']
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    x=np.arange(2);ax=axes[0,0]
    ax.bar(x-.18,[costs[m]['work'] for m in runs],.35,label='Work (terms)')
    ax.bar(x+.18,[costs[m]['largest_tensor'] for m in runs],.35,label='Largest tensor (entries)')
    ax.set(xticks=x,xticklabels=list(runs),yscale='log',title='Exact scalar-query plan costs');ax.legend()
    ax=axes[0,1];x=np.arange(3)
    ax.bar(x-.2,mar[ids,1],.2,label='Published MAR');ax.bar(x,[runs['treesa']['selected_marginals'][str(q)][1] for q in ids],.2,label='Exact');ax.bar(x+.2,bp_selected[:,1],.2,label='BP')
    ax.set(xticks=x,xticklabels=ids,xlabel='Variable ID',ylabel='State-1 probability',title='BP remains inaccurate');ax.legend()
    ax=axes[1,0];ax.semilogy(sweeps,residual);ax.axhline(bp['tolerance'],ls='--',color='gray',label='Stopping tolerance');ax.legend();ax.set(xlabel='Completed sweeps',ylabel='Undamped residual',title='BP: iteration limit reached, not converged')
    ax=axes[1,1];ax.plot(sweeps,errors,color='#c87135');ax.set(xlabel='Completed sweeps',ylabel='Maximum selected-query error vs MAR',title='Convergence and accuracy are separate')
    fig.tight_layout();fig.savefig(ROOT/'evidence/promedus.png',dpi=170);plt.close(fig)
    (ROOT/'evidence/promedus.json').write_text(json.dumps(checks,indent=2)+'\n')
    # Retain all node beliefs, including the returned state, without a 32 MB text file.
    for case in ['prior','AD','ADX']:
        path=ROOT/f'artifacts/bp-{case}/run.json';(ROOT/f'evidence/bp-{case}.json.gz').write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    path=ROOT/'artifacts/promedus-bp/run.json';(ROOT/'evidence/promedus-bp.json.gz').write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
