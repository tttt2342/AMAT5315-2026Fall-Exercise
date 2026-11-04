"""Verify saved coursework evidence against independent sums and published data."""
import gzip
import hashlib
import itertools
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from plot_orders import recount
from plot_benchmark import published
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def main():
    tests=subprocess.run(['cargo','+1.96.0','test','--release','--locked','--offline'],cwd=ROOT,capture_output=True,text=True)
    print(tests.stdout);print(tests.stderr);assert tests.returncode==0
    (ROOT/'evidence/rust_tests.txt').write_text(tests.stdout)
    # Independent UAI parsing and direct summation in file scope order.
    words=iter((ROOT/'asia.uai').read_text().split());assert next(words)=='MARKOV'
    n=int(next(words));dims=[int(next(words)) for _ in range(n)];nf=int(next(words))
    scopes=[[int(next(words)) for _ in range(int(next(words)))] for _ in range(nf)]
    arrays=[]
    for scope in scopes:
        k=int(next(words));assert k==int(np.prod([dims[v] for v in scope]))
        arrays.append(np.array([float(next(words)) for _ in range(k)]).reshape([dims[v] for v in scope]))
    assert arrays[1].tolist()==[[.05,.01],[.95,.99]]
    evidence=read('evidence/asia.json');checks={}
    for label,obs,zref in [('prior',{},1.),('AD',{0:0,7:0},.004501375),('ADX',{0:0,7:0,6:0},.00098822675)]:
        joint=np.zeros(dims)
        for x in itertools.product(*[range(d) for d in dims]):
            if all(x[v]==s for v,s in obs.items()):joint[x]=np.prod([f[tuple(x[v] for v in scope)] for scope,f in zip(scopes,arrays)])
        z=float(joint.sum());assert abs(z/zref-1)<1e-12;joint/=z
        nodes=[joint.sum(axis=tuple(i for i in range(n) if i!=v)) for v in range(n)]
        tl=joint.sum(axis=(0,2,4,5,6,7));run=evidence[label]
        err=max(np.max(abs(np.array(nodes)-run['node_marginals'])),np.max(abs(tl.ravel()-run['joint_TL'])))
        assert err<1e-10;assert abs(run['evidence_mass']/z-1)<1e-10
        assert all(abs(sum(p)-1)<1e-12 for p in run['node_marginals']);assert abs(sum(run['joint_TL'])-1)<1e-12
        checks[label]={'numpy_marginal_error':float(err),'mass':z}
    for method in ['random','greedy']:
        recount(read(f'runs/{method}/order.json'),read(f'runs/{method}/complexity.json'))
    assert read('runs/analysis/complexity.json')==read('runs/random/complexity.json')
    assert abs(read('runs/replay/run.json')['evidence_mass']/.004501375-1)<1e-10
    pr,mar=published();exact=[]
    for method in ['greedy','treesa']:
        prefix=f'runs/promedus-{method}/';run=read(prefix+'run.json');exact.append(run)
        recount(read(prefix+'order.json'),read(prefix+'complexity.json'))
        assert abs(run['log10_partition']-pr)<1e-4
        for q in [59,102,139]:
            recount(read(prefix+f'query-{q}-order.json'),read(prefix+f'query-{q}-complexity.json'))
            assert max(abs(np.array(run['selected_marginals'][str(q)])-mar[q]))<1e-5
    assert abs(exact[0]['log10_partition']-exact[1]['log10_partition'])<1e-10
    for q in [59,102,139]:assert max(abs(np.array(exact[0]['selected_marginals'][str(q)])-exact[1]['selected_marginals'][str(q)]))<1e-10
    bp_status={}
    for case in ['bp-prior','bp-AD','bp-ADX','promedus-bp']:
        run=json.loads(gzip.decompress((ROOT/f'evidence/{case}.json.gz').read_bytes()))
        assert len(run['history'])==run['sweeps']+1
        assert run['history'][-1]['node_beliefs']==run['node_marginals']
        assert run['history'][-1]['residual']==run['residual']
        assert run['converged']==(run['residual']<run['tolerance'])
        for row in run['history']:
            p=np.array(row['node_beliefs']);assert np.max(abs(p.sum(axis=1)-1))<1e-12;assert np.min(p)>=0
        bp_status[case]={'converged':run['converged'],'residual':run['residual'],'sweeps':run['sweeps']}
    assert not bp_status['promedus-bp']['converged']
    chinese=[]
    for path in ROOT.rglob('*'):
        if any(part in {'.venv','target','artifacts','bin','__pycache__'} for part in path.relative_to(ROOT).parts):continue
        if path.suffix in {'.rs','.py','.md','.toml','.json','.svg','.sh','.txt','.lock'} and re.search(r'[\u3400-\u9fff]',path.read_text()):chinese.append(str(path))
    assert not chinese,chinese
    files=[ROOT/'week9-learning-sheet.pdf',ROOT/'asia.uai',ROOT/'Cargo.lock',ROOT/'uv.lock',*sorted((ROOT/'promedus').glob('*')),*sorted((ROOT/'src').glob('*.rs')),*sorted((ROOT/'evidence').glob('*.png'))]
    versions={'rust':subprocess.check_output(['rustc','+1.96.0','--version'],text=True).strip(),'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__}
    import matplotlib
    versions['matplotlib']=matplotlib.__version__
    result={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'rust_tests_passed':7,'asia_independent_numpy':checks,'bp_status':bp_status,'plans_independently_recounted':True,'published_PR_tolerance':1e-4,'published_MAR_tolerance':1e-5,'exact_plan_tolerance':1e-10,'new_text_files_contain_CJK':False,'versions':versions,'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (ROOT/'evidence/verification.json').write_text(json.dumps(result,indent=2)+'\n');print('All implementation and exact-inference checks passed. BP limitations remain explicitly reported.')
if __name__=='__main__':main()
