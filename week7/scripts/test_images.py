import unittest,json
import numpy as np
import jax
import jax.numpy as jnp
from mixture import AB,A,E
from images import repaint

class RepaintTests(unittest.TestCase):
    def test_algorithm_against_independent_numpy_loop(self):
        # A small deterministic denoiser makes the entire sampler cheaply testable.
        obs=np.arange(48,dtype=np.float32).reshape(1,3,4,4)/48-.5
        mask=np.ones_like(obs,dtype=bool);mask[:,:,1:3,1:3]=False;obs=np.where(mask,obs,0)
        fake=lambda w,x,t:.1*x
        actual,count=repaint(fake,{},jnp.array(obs),jnp.array(mask),42)
        self.assertEqual(count,99)
        key=jax.random.PRNGKey(42)
        random=lambda draw:np.asarray(jax.random.normal(jax.random.fold_in(key,draw),obs.shape),dtype=np.float64)
        x=np.asarray(jax.random.normal(key,obs.shape),dtype=np.float64);ab=np.asarray(AB,dtype=np.float64);draw=1
        for t in range(1000,0,-20):
            tp=t-20;a=ab[t]/ab[tp]
            for r in range(2 if tp else 1):
                if r:x=np.sqrt(a)*x+np.sqrt(1-a)*random(draw);draw+=1
                clean=np.clip((x-np.sqrt(1-ab[t])*.1*x)/np.sqrt(ab[t]),-1,1)
                mu=np.sqrt(ab[tp])*(1-a)/(1-ab[t])*clean+np.sqrt(a)*(1-ab[tp])/(1-ab[t])*x
                variance=(1-ab[tp])*(1-a)/(1-ab[t])
                proposal=mu+np.sqrt(variance)*random(draw)
                known=np.sqrt(ab[tp])*obs+np.sqrt(1-ab[tp])*random(draw+1);draw+=2
                x=np.where(mask,known,proposal)
        np.testing.assert_allclose(actual,x,atol=3e-5,rtol=3e-5)
        np.testing.assert_array_equal(actual[mask],obs[mask])
    def test_unet_reference(self):
        r=json.loads((E/'unet_verification.json').read_text())
        self.assertEqual(len(r['checks']),4)
        self.assertLess(max(x['relative_rms'] for x in r['checks']),1e-5)
    def test_saved_photographs(self):
        data=np.load(A/'photographs.npz')
        self.assertEqual(data['t0'].shape,(4,3,32,32))
        self.assertEqual(data['t0'].dtype,np.float32)
        self.assertTrue(np.isfinite(data['t0']).all())
        r=json.loads((E/'photographs.json').read_text());self.assertEqual(r['steps'],1000)
    def test_saved_inpainting(self):
        data=np.load(A/'inpainting.npz');c=data['completions'];mask=data['mask'];truth=data['truth']
        self.assertEqual(c.shape,(4,4,3,32,32));self.assertEqual(np.sum(~mask),4*1200)
        np.testing.assert_array_equal(c[:,mask],np.broadcast_to(truth[mask],c[:,mask].shape))
        self.assertGreater(float(c.var(0)[~mask].mean()),0)
        r=json.loads((E/'inpainting.json').read_text());self.assertEqual(r['network_evaluations_per_seed'],[99]*4)
        measured=((c-truth)**2)[:,:,:,6:26,6:26].mean(axis=(2,3,4))
        np.testing.assert_allclose(measured,r['mse_seed_by_image'])

if __name__=='__main__':unittest.main()
