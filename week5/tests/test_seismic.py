"""Independent finite differences, arbitrary transpose weights, schedule endpoints."""
import json
import subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BIN=ROOT/'seismic/target/release/seismic'
def run(tmp,e,mode,name,extra=()):
    p=tmp/f'{name}.json';p.write_text(json.dumps(e));out=tmp/name
    subprocess.run([str(BIN),'--experiment',str(p),'--mode',mode,'--out',str(out),*map(str,extra)],check=True,capture_output=True)
    return out

def experiment():
    e=json.loads((ROOT/'inputs/reflector.json').read_text())
    e.update(nx=9,nz=8,steps=13,dt=.1,shots=[[4,3]],receivers=[[3,3],[3,3],[5,3]],source_peak_time=.2,source_frequency=.7,sponge_width=2)
    e['background']=np.full((8,9),1.8).tolist()
    e['perturbation']=(np.random.default_rng(53).normal(size=(8,9))*.1).tolist()
    return e

def test_trajectory_born_finite_difference_and_arbitrary_transpose(tmp_path):
    e=experiment();m=np.array(e['perturbation']);c=np.array(e['background'])
    born=np.load(run(tmp_path,e,'born','born')/'born_data.npy');h=1e-4
    fields=[]
    for sign in [1,-1]:
        f=dict(e,background=(c+sign*h*m).tolist())
        fields.append(np.load(run(tmp_path,f,'forward',f'fd{sign}')/'traces.npy'))
    assert np.linalg.norm((fields[0]-fields[1])/(2*h)-born)/np.linalg.norm(born)<1e-7
    w=np.random.default_rng(12).normal(size=born.shape);p=tmp_path/'weights.npy';np.save(p,w)
    full=np.load(run(tmp_path,e,'adjoint','full',['--data',p])/'image.npy')
    lhs=np.sum(w*born);rhs=np.sum(full*m)
    assert abs(lhs-rhs)/max(abs(lhs),abs(rhs))<1e-11
    for b in [1,2,5]:
        cp=np.load(run(tmp_path,e,'adjoint',f'cp{b}',['--data',p,'--storage','treeverse','--checkpoints',b])/'image.npy')
        np.testing.assert_array_equal(cp,full)

def test_six_step_treeverse_example_and_single_step(tmp_path):
    for n in [1,6]:
        e=experiment();e['steps']=n;p=tmp_path/f'w{n}.npy';np.save(p,np.ones((1,n,3)))
        out=run(tmp_path,e,'adjoint',f'n{n}',['--data',p,'--storage','treeverse','--checkpoints',2]);a=json.loads((out/'actions-0.json').read_text());stats=json.loads((out/'result.json').read_text())['statistics']
        assert [r['step'] for r in a if r['action']=='grad']==list(range(n-1,-1,-1))
        assert stats['scheduler_forward_calls']==(8 if n==6 else 0)
        assert stats['peak_saved_states']==(3 if n==6 else 1)
