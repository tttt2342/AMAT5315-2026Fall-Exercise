"""Attention equality, energy descent and full retrieval success checks."""
import json
import unittest
import jax.numpy as jnp
import numpy as np
from scipy.special import softmax
from data import ARTIFACTS
from modern import modern_update,attention_update,modern_energy,retrieval_success

class ModernTests(unittest.TestCase):
    def test_attention_matches_numpy_weighted_average(self):
        rng=np.random.default_rng(51)
        patterns=rng.choice([-1.,1.],size=(13,16)).astype(np.float32)
        query=rng.normal(size=(7,16)).astype(np.float32)
        for beta in [.001,.03,.3,1.]:
            expected=softmax(beta*(query@patterns.T),axis=1)@patterns
            actual,weights=modern_update(jnp.array(patterns),jnp.array(query),beta)
            attention=attention_update(jnp.array(patterns),jnp.array(query),beta)
            np.testing.assert_allclose(actual,expected,atol=1e-6)
            np.testing.assert_allclose(attention,expected,atol=1e-6)
            self.assertTrue(np.all(np.array(modern_energy(jnp.array(patterns),actual,beta))<=np.array(modern_energy(jnp.array(patterns),jnp.array(query),beta))+1e-5))

    def test_success_requires_target_to_be_nearest(self):
        patterns=np.ones((2,100),np.float32); patterns[1,0]=-1
        # Overlap .98 alone would pass, but the retrieved output is the other stored pattern.
        success,overlap,_=retrieval_success(patterns[1:2],patterns,np.array([0]))
        self.assertGreaterEqual(overlap[0],.95)
        self.assertFalse(success[0])

    def test_full_grid_and_numpy_attention(self):
        summary=json.loads((ARTIFACTS/'modern.json').read_text())
        self.assertEqual(summary['P'],[3,10,30,100,300,1000])
        self.assertEqual(summary['beta'],[.001,.003,.01,.03,.1,.3,1.])
        self.assertLess(summary['max_attention_difference'],1e-5)
        for p in summary['P']:
            stats=summary['models'][str(p)]
            self.assertEqual(stats['classical'],0)
            self.assertGreaterEqual(stats['modern'][5],.9)
            with np.load(ARTIFACTS/f'modern_P{p}.npz') as f:
                np.testing.assert_array_equal(f['target_ids'],np.arange(100)%p)
                np.testing.assert_array_equal(f['queries'][:,:392],f['targets'][:,:392])
                for i,beta in enumerate(summary['beta']):
                    # Float64 NumPy check is independent of the JAX attention implementation.
                    patterns=f['patterns'].astype(np.float64); query=f['queries'].astype(np.float64)
                    expected=softmax(beta*(query@patterns.T),axis=1)@patterns
                    # float32 reduction over up to 1000 stored patterns; use the sheet's 1e-5 bound.
                    np.testing.assert_allclose(f['modern'][i],expected,atol=1e-5,rtol=0)

if __name__=='__main__': unittest.main(verbosity=2)
