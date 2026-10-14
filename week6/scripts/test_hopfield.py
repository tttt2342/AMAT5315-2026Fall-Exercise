"""NumPy reference trajectory, energy identity, zero-field rule, capacity acceptance."""
import json
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from data import ARTIFACTS
from hopfield import hebbian,recall_one,energy

class HopfieldTests(unittest.TestCase):
    def test_hebbian_and_energy_identity(self):
        rng=np.random.default_rng(44)
        patterns=rng.choice([-1,1],size=(5,16)).astype(np.int32)
        matrix=np.array(hebbian(patterns))
        expected=patterns.T@patterns; np.fill_diagonal(expected,0)
        np.testing.assert_array_equal(matrix,expected)
        state=rng.choice([-1,1],size=16).astype(np.int32)
        overlap=patterns@state/16
        self.assertAlmostEqual(float(energy(jnp.array(matrix),jnp.array(state))),-8*np.sum(overlap**2)+2.5)

    def test_asynchronous_trajectory_against_numpy(self):
        rng=np.random.default_rng(442)
        patterns=rng.choice([-1,1],size=(12,32)).astype(np.int32)
        matrix=patterns.T@patterns; np.fill_diagonal(matrix,0)
        initial=patterns[0].copy(); initial[:8]*=-1
        original_key=jax.random.key(17); key=original_key; state=initial.copy(); sweeps=0
        while True:
            key,order_key=jax.random.split(key)
            order=np.array(jax.random.permutation(order_key,len(state)))
            flips=0
            for i in order:
                field=matrix[i]@state
                new=1 if field>0 else (-1 if field<0 else state[i])
                before=-state@matrix@state/2
                delta=new-state[i]; state[i]=new
                after=-state@matrix@state/2
                self.assertLessEqual(after,before)
                if delta: self.assertLess(after,before)
                flips+=int(delta!=0)
            sweeps+=1
            if not flips: break
        actual,count=jax.jit(recall_one)(jnp.array(matrix),jnp.array(initial),original_key)
        np.testing.assert_array_equal(actual,state)
        self.assertEqual(int(count),sweeps)
        self.assertTrue(np.all(state*(matrix@state)>=0))

    def test_zero_field_keeps_spin(self):
        initial=jnp.array([-1,1,-1,1],dtype=jnp.int32)
        final,count=jax.jit(recall_one)(jnp.zeros((4,4),dtype=jnp.int32),initial,jax.random.key(0))
        np.testing.assert_array_equal(final,initial)
        self.assertEqual(int(count),1)

    def test_capacity_full_budget_and_acceptance(self):
        s=json.loads((ARTIFACTS/'hopfield.json').read_text())
        self.assertEqual(s['sets_per_point'],20)
        np.testing.assert_allclose(s['loads'],np.arange(5,26)/100)
        values=[]
        for n in [100,200,400,800]:
            values.append(s['models'][str(n)]['crossing'])
            for alpha in s['loads']:
                p=round(n*alpha)
                with np.load(ARTIFACTS/f'hopfield_N{n}_P{p}.npz') as f:
                    self.assertEqual(f['patterns'].shape,(20,p,n))
                    self.assertEqual(f['overlap'].shape,(20,min(p,10)))
                    # Independently recompute unstable fractions with integer NumPy arithmetic.
                    patterns=f['patterns'][0].astype(np.int32)
                    matrix=patterns.T@patterns; np.fill_diagonal(matrix,0)
                    self.assertAlmostEqual(float(f['unstable'][0]),np.mean(patterns*(patterns@matrix)<0),places=7)
        self.assertTrue(all(a>b for a,b in zip(values,values[1:])))
        self.assertTrue(np.all(np.abs(np.array(values)-[.186,.170,.156,.150])<=.02))

if __name__=='__main__': unittest.main(verbosity=2)
