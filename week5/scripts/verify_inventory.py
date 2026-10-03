"""Validate the sheet's complete inventory, including ignored local arrays."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
root=Path('artifacts')
expected_png=['inputs.png','ad/modes.png','ad/graph.png','ad/grad-graph.png','ad/scaling.png','forward/wavefield.png','forward/echo.png','forward/gathers.png','adjoint/wavefield.png','adjoint/image.png','checkpoint-actions.png','checkpoint-work.png','marmousi.png']
for name in expected_png:
    with Image.open(root/name) as im: im.verify()
assert (root/'ad/derivatives.json').is_file()
for folder in ['forward','born','adjoint',*[f'checkpoint-{b}' for b in [1,3,5,10]],'marmousi-born','marmousi-image']:
    p=root/folder;r=json.loads((p/'run.json').read_text());s=json.loads((p/'result.json').read_text());e=r['experiment'];assert 'background' not in e and 'perturbation' not in e
    l=e['nx']*e['nz'];shape=(len(e['shots']),e['steps'],len(e['receivers']));mode=s['mode']
    name={'forward':'traces.npy','born':'born_data.npy','adjoint':'image.npy'}[mode];a=np.load(p/name);assert a.dtype==np.float64 and np.isfinite(a).all();assert a.shape==((e['nz'],e['nx']) if mode=='adjoint' else shape)
    if 'recording' in r:
        for name in (['wavefield.npy','echo.npy'] if mode=='forward' else ['wavefield.npy']):
            a=np.load(p/name);assert a.dtype==np.float32 and np.isfinite(a).all();assert a.shape==(len(r['recording']['steps']),e['nz'],e['nx'])
        assert np.allclose(r['recording']['times'],np.array(r['recording']['steps'])*e['dt'])
    if mode=='adjoint':
        st=s['statistics'];assert st['reverse_calls']==len(e['shots'])*e['steps'];assert st['peak_saved_bytes']==st['peak_saved_states']*l*16
        if st['storage']=='treeverse':
            assert len(st['per_shot'])==len(e['shots']);assert all((p/shot['actions_file']).is_file() for shot in st['per_shot'])
        else: assert st['peak_saved_states']==e['steps']+1
files=[]
for p in sorted(root.rglob('*')):
    if not p.is_file() or p.suffix=='.npy': continue
    assert p.stat().st_size<5_000_000,(p,p.stat().st_size)
    if p.suffix=='.json': json.loads(p.read_text())
    files.append(str(p))
print(f'PASS: {len(expected_png)} PNGs, all run/array contracts, JSON parsing, and every committed evidence file < 5 MB')
print('\n'.join(files))
