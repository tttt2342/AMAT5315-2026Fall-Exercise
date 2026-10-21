"""Run the focused tests and record the environment and artifact provenance."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
from pathlib import Path
import sys,json,hashlib,platform,subprocess
from importlib.metadata import version
import jax
from mixture import ROOT,A,E

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

if __name__=='__main__':
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'scripts'),'-p','test_*.py','-v'],capture_output=True,text=True)
    print(result.stdout+result.stderr)
    (A/'tests.log').write_text(result.stdout+result.stderr)
    downloads=[ROOT/'week7-learning-sheet.pdf',ROOT/'inputs/images.npz',ROOT/'inputs/cifar-model/config.json',ROOT/'inputs/cifar-model/diffusion_pytorch_model.safetensors']
    evidence=sorted(list(E.glob('*.png'))+list(E.glob('*.gif')))
    report=dict(python=platform.python_version(),platform=platform.platform(),backend=jax.default_backend(),packages={p:version(p) for p in ['jax','jaxlib','optax','numpy','scipy','matplotlib','pillow','safetensors','huggingface-hub','torch','diffusers']},test_returncode=result.returncode,test_count=11,downloads={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in downloads},figures={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in evidence},model_revision='267b167dc01f0e4e61923ea244e8b988f84deb80',raw_files=[p.name for p in sorted(A.glob('*.npz'))],known_reference_difference='Part 2 learned DDIM width range 0.27793–0.31634; reference 0.29–0.32; retained without seed selection.')
    (E/'verification.json').write_text(json.dumps(report,indent=2))
    settings=dict(part1=dict(samples=2000,forward_seed=7110,start_seed=7111,reverse_seed=7112,stage_reverse_seed=7113),part2=dict(start='part1 same start',reverse_base_key=7120,stream_ids={'DDIM A':0,'DDIM B':1,'DDPM A':2,'DDPM B':3},exact_target_seed=7121),part3=dict(samples=4000,steps=[1,2,5,10,20,50,100],additional_diffusion_steps=[1000],start_seed=7130,target_seed=7131,null_pairs=20,null_seeds=list(range(7140,7180)),reverse_seed=7190,velocity_data_seed=7181,velocity_noise_seed=7182,velocity_samples=10000),part4=dict(seed=7200,steps=1000,sample_count=4,reference_input_seed=7240),part5=dict(seeds=[7201,7202,7203,7204],input_key='test',mask='all channels, rows and columns 6:26',levels=list(range(1000,-1,-20)),attempts='2 except final 1',network_evaluations=99))
    (A/'run_settings.json').write_text(json.dumps(settings,indent=2));(E/'run_settings.json').write_text(json.dumps(settings,indent=2))
    sys.exit(result.returncode)
