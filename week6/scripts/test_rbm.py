"""Independent small-system checks and required full-run acceptance tests."""
import itertools
import json
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit,logsumexp
from data import ARTIFACTS,mnist
from rbm import energy,free_energy,surrogate,gibbs_step

class RBMTests(unittest.TestCase):
    def setUp(self):
        self.params=tuple(jnp.array(x,dtype=jnp.float32) for x in
            ([[.2,-.3],[.7,.1],[-.4,.5]],[.1,-.2,.3],[-.1,.2]))

    def test_free_energy_by_exact_hidden_enumeration(self):
        w,a,b=map(np.array,self.params)
        hs=np.array(list(itertools.product([0,1],repeat=2)))
        for v in itertools.product([0,1],repeat=3):
            v=np.array(v,np.float32)
            energies=np.array([-a@v-b@h-v@w@h for h in hs])
            np.testing.assert_allclose(energy(self.params,jnp.array(v),jnp.array(hs)),energies,atol=2e-7)
            self.assertAlmostEqual(float(free_energy(self.params,jnp.array(v))),-logsumexp(-energies),places=6)

    def test_cd_gradient_equals_correlation_difference(self):
        data=np.array([[1,0,1],[0,1,0]],np.float32)
        endpoint=np.array([[1,1,0],[0,0,1]],np.float32)
        w,a,b=map(np.array,self.params)
        ph0=expit(data@w+b); phk=expit(endpoint@w+b)
        expected=((endpoint.T@phk-data.T@ph0)/2,(endpoint-data).mean(0),(phk-ph0).mean(0))
        actual=jax.grad(surrogate)(self.params,jnp.array(data),jnp.array(endpoint))
        for x,y in zip(actual,expected): np.testing.assert_allclose(x,y,atol=1e-7)
        np.testing.assert_array_equal(jax.grad(surrogate,argnums=2)(self.params,jnp.array(data),jnp.array(endpoint)),0)

    def test_sampled_gibbs_conditional_marginal(self):
        v=np.array([1,0,1],np.float32)
        w,a,b=map(np.array,self.params)
        ph=expit(v@w+b)
        expected=np.zeros(3)
        for h in itertools.product([0,1],repeat=2):
            h=np.array(h); probability=np.prod(np.where(h,ph,1-ph))
            expected+=probability*expit(h@w.T+a)
        draws=np.array(jax.jit(gibbs_step)(self.params,jnp.tile(v,(50000,1)),jax.random.key(31)))
        self.assertTrue(np.isin(draws,[0,1]).all())
        np.testing.assert_allclose(draws.mean(0),expected,atol=.01)

    def test_full_run_acceptance_and_clamping(self):
        hist=json.loads((ARTIFACTS/'rbm_training.json').read_text())
        self.assertEqual(hist['settings']['training']['images'],60000)
        self.assertEqual(hist['models']['1']['shuffle_sha256'],hist['models']['10']['shuffle_sha256'])
        starts=[]; labels=mnist('test')[1][:64]
        for k in (1,10):
            model=hist['models'][str(k)]
            self.assertEqual(len(model['surrogate']),20)
            self.assertLess(model['reconstruction'][-1],model['reconstruction'][0]/2)
            with np.load(ARTIFACTS/f'cd{k}_generation.npz') as f: starts.append(f['frames'][0])
            with np.load(ARTIFACTS/f'cd{k}_completion.npz') as f:
                frames=f['frames']; target=f['targets']; wrong=f['wrong_indices']
                self.assertEqual(frames.shape,(5,2,64,8,784))
                for i,j in enumerate(wrong):
                    self.assertNotEqual(labels[i],labels[j])
                    self.assertTrue(all(labels[(i+d)%64]==labels[i] for d in range(1,(j-i)%64)))
                for condition,observed in enumerate([target,target[wrong]]):
                    np.testing.assert_array_equal(frames[:,condition,:,:,:392],np.broadcast_to(observed[None,:,None,:392],(5,64,8,392)))
                advantage=f['dice'][-1,0].mean()-f['dice'][-1,1].mean()
                self.assertGreaterEqual(advantage,.1)
        np.testing.assert_array_equal(starts[0],starts[1])

if __name__=='__main__': unittest.main(verbosity=2)
