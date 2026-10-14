"""Run scientific tests and retain a compact, versioned acceptance report."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MPLCONFIGDIR','/tmp/amat5315-mpl')
import io
import json
import platform
import unittest
from datetime import datetime,timezone
import jax
import numpy as np
import scipy
from scipy.special import softmax
from data import ROOT,ARTIFACTS,EVIDENCE


def main():
    suite=unittest.defaultTestLoader.discover(str(ROOT/'scripts'),pattern='test_*.py')
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue())
    (ARTIFACTS/'tests.log').write_text(stream.getvalue())
    if not result.wasSuccessful(): raise SystemExit(1)
    rbm=json.loads((EVIDENCE/'rbm_training.json').read_text())
    completion=json.loads((EVIDENCE/'rbm_evaluation.json').read_text())
    hopfield=json.loads((EVIDENCE/'hopfield.json').read_text())
    modern=json.loads((EVIDENCE/'modern.json').read_text())
    difference=0.
    for p in modern['P']:
        with np.load(ARTIFACTS/f'modern_P{p}.npz') as f:
            patterns=f['patterns'].astype(np.float64); queries=f['queries'].astype(np.float64)
            for bi,beta in enumerate(modern['beta']):
                expected=softmax(beta*(queries@patterns.T),axis=1)@patterns
                difference=max(difference,float(np.max(np.abs(f['modern'][bi]-expected))))
    figures=sorted(p.name for p in EVIDENCE.glob('*.png'))
    report=dict(verified_utc=datetime.now(timezone.utc).isoformat(),tests_run=result.testsRun,tests_passed=True,
        environment=dict(python=platform.python_version(),platform=platform.platform(),machine=platform.machine(),
                         jax=jax.__version__,numpy=np.__version__,scipy=scipy.__version__,backend=jax.default_backend(),devices=list(map(str,jax.devices()))),
        rbm_final_reconstruction={k:v['reconstruction'][-1] for k,v in rbm['models'].items()},
        completion_advantage={k:v['final_advantage'] for k,v in completion['models'].items()},
        capacity_crossings={k:v['crossing'] for k,v in hopfield['models'].items()},
        modern_beta_03_success={k:v['modern'][5] for k,v in modern['models'].items()},
        classical_success={k:v['classical'] for k,v in modern['models'].items()},
        attention_max_difference=modern['max_attention_difference'],numpy_float64_attention_max_difference=difference,
        figure_count=len(figures),figures=figures)
    assert difference<1e-5
    (EVIDENCE/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
