"""Separate fixed-point convergence from marginal accuracy."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
    fig,axes=plt.subplots(2,3,figsize=(12,6));summary={}
    for j,case in enumerate(['prior','AD','ADX']):
        run=json.loads((ROOT/f'artifacts/bp-{case}/run.json').read_text())
        exact=np.array(json.loads((ROOT/f'artifacts/{case}/run.json').read_text())['node_marginals'])
        hist=run['history'];errors=np.array([np.max(np.abs(np.array(h['node_beliefs'])-exact)) for h in hist])
        residuals=[h['residual'] for h in hist];sweeps=[h['sweep'] for h in hist]
        axes[0,j].semilogy(sweeps,residuals);axes[0,j].axhline(run['tolerance'],ls='--',color='gray');axes[0,j].set(title=case,ylabel='Undamped residual',xlabel='Completed sweeps')
        axes[1,j].semilogy(sweeps,np.maximum(errors,1e-16),color='#c87135');axes[1,j].set(ylabel='Maximum node marginal error',xlabel='Completed sweeps')
        summary[case]={k:run[k] for k in ['sweeps','residual','converged','node_marginals','runtime_seconds','cumulative_update_work']};summary[case]['max_marginal_error']=float(errors[-1])
    fig.suptitle('ASIA belief propagation: a small residual does not imply accuracy');fig.tight_layout();fig.savefig(ROOT/'evidence/bp.png',dpi=170)
    (ROOT/'evidence/bp_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for c,r in summary.items():print(c,r['sweeps'],r['residual'],r['max_marginal_error'])
if __name__=='__main__':main()
