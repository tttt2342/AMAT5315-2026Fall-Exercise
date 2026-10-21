"""Independent formula checks and checks against saved full-budget experiments."""
import unittest, json
import numpy as np
import jax
import jax.numpy as jnp
from mixture import AB,BETA,CENTRES,A,E,exact_noise,exact_velocity,energy,sample,load

class FormulaTests(unittest.TestCase):
    def test_schedule(self):
        beta=np.linspace(.0001,.02,1000,dtype=np.float64)
        np.testing.assert_allclose(AB,np.r_[1,np.cumprod(1-beta)],atol=3e-7,rtol=2e-6)
        self.assertEqual(float(AB[0]),1.)
    def test_exact_fields_against_gaussian_conditioning(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(30,2));c=np.asarray(CENTRES)
        for t in [1,100,500,1000]:
            a=float(AB[t]);variance=.09*a+1-a;d=x[:,None]-np.sqrt(a)*c
            logits=-(d*d).sum(-1)/(2*variance);weights=np.exp(logits-logits.max(1,keepdims=True));weights/=weights.sum(1,keepdims=True)
            posterior=c+.09*np.sqrt(a)/variance*d
            clean=(weights[:,:,None]*posterior).sum(1)
            expected=(x-np.sqrt(a)*clean)/np.sqrt(1-a)
            np.testing.assert_allclose(exact_noise(x,t),expected,atol=2e-5,rtol=2e-5)
        for s in [.1,.5,.9,1.]:
            var=.09*(1-s)**2+s*s;d=x[:,None]-(1-s)*c
            logits=-(d*d).sum(-1)/(2*var);weights=np.exp(logits-logits.max(1,keepdims=True));weights/=weights.sum(1,keepdims=True)
            ez=s/var*d;ex=c+.09*(1-s)/var*d
            expected=(weights[:,:,None]*(ez-ex)).sum(1)
            np.testing.assert_allclose(exact_velocity(x,s),expected,atol=3e-6,rtol=3e-6)
    def test_energy_unbiased_pairs(self):
        x=np.array([[0.,0.],[2.,0.]]);y=np.array([[1.,0.],[4.,0.]])
        self.assertAlmostEqual(energy(x,y),2*(1+4+1+2)/4-2-3)
    def test_one_step_final_has_no_noise(self):
        p=load('noise');x=jnp.array([[1.,2.],[-1.,.5]])
        a=sample(p,x,jax.random.PRNGKey(1),steps=1)
        b=sample(p,x,jax.random.PRNGKey(2),steps=1)
        np.testing.assert_array_equal(a,b)
        # Exact flow at s=1 is x: a single Euler step collapses to zero.
        y=sample(p,x,jax.random.PRNGKey(1),method='flow',steps=1,analytic=True)
        np.testing.assert_allclose(y,0,atol=1e-6)

class EvidenceTests(unittest.TestCase):
    def test_part1(self):
        r=json.loads((E/'part1.json').read_text())
        self.assertLess(max(abs(np.array(r['forward_mean']))),.1)
        self.assertLess(max(abs(np.array(r['forward_variance'])-1)),.1)
        for field,tol in [('fraction_error',.04),('centre_l2_error',.05),('width_error',.03)]:self.assertLess(r['ddpm'][field],tol)
        self.assertEqual(np.load(A/'part1.npz')['end'].shape,(2000,2))
    def test_part2(self):
        r=json.loads((E/'part2.json').read_text());self.assertEqual(r['ddim_max_difference'],0)
        self.assertLess(r['DDIM A']['fraction_error'],.015)
        for name in ['DDIM A','DDPM A','DDPM B']:
            # The 0.29–0.32 reference range is recorded separately, not hidden.
            self.assertLess(r[name]['width_error'],.03)
        actual=all(.29 <= np.array(r[n]['widths']).min() and np.array(r[n]['widths']).max() <= .32 for n in ['DDIM A','DDPM A','DDPM B'])
        self.assertEqual(r['reference_width_range_matched'],actual)
    def test_part3(self):
        r=json.loads((E/'part3.json').read_text());self.assertEqual(r['sample_count'],4000);self.assertEqual(len(r['null_distances']),20)
        self.assertLess(max(r['velocity_relative_rms'].values()),.1)
        self.assertLess(r['flow_learned_20']['energy_distance'],.005);self.assertLess(r['flow_learned_20']['width_error'],.03)
        for method in ['ddpm','ddim','flow']:
            for n in [1,2,5,10,20,50,100]+([] if method=='flow' else [1000]):self.assertIn(f'{method}_learned_{n}',r)

if __name__=='__main__':unittest.main()
