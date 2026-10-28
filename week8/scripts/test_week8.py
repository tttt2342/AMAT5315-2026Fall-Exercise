import unittest
from common import *
from counts import codes,fit,score,lookup
from neural import *
from generate import distribution
from scaling import objective,optimum,law
from memory import ratios,capacity
from scipy.special import softmax

class FormulaChecks(unittest.TestCase):
 def test_memory_signal_and_capacity(self):
  keys=np.eye(4); values=np.eye(4)
  np.testing.assert_allclose(ratios(keys,values),0,atol=1e-12)
  self.assertAlmostEqual(capacity(np.array([8,16,32]),np.array([.5,1.,2.])),16.)
  self.assertAlmostEqual(capacity(np.array([8,32]),np.array([.5,2.])),16.)
 def test_count_against_hand_counts(self):
  t=np.array([1,2,1,2,2]); m=fit(t,2,2); p=(np.array([0,1,1])+0.1)/(2+512*.1)
  q=lookup(m[0],m[1],np.array([2*512+1,1*512+2,2*512+2],dtype=np.uint64))
  np.testing.assert_array_equal(q,[2,2,1]); self.assertAlmostEqual(score(m,t,2,.1,2)[0],-np.mean(np.log(np.array([2.1/54.2,2.1/53.2,2.1/54.2,2.1/53.2,1.1/54.2]))))
 def test_attention_numpy_and_permutation(self):
  p=init_model('recall',1); x,y,_=recall_batch(np.random.default_rng(3),4); actual,a=recall(p,x,True); p=jax.tree.map(np.asarray,p); h=p['e'][x]
  q,k,v=[h@p[z]['w']+p[z]['b'] for z in ['q','k','v']]; weights=softmax(q@k.swapaxes(-1,-2)/8,axis=-1); z=h+(weights@v)@p['o']['w']+p['o']['b']; z=(z-z.mean(-1,keepdims=True))/np.sqrt(z.var(-1,keepdims=True)+1e-5)*p['ln']['g']+p['ln']['b']; out=z@p['out']['w']+p['out']['b']
  np.testing.assert_allclose(actual,out[:,-1],rtol=2e-5,atol=2e-6)
  perm=x[:,[2,0,3,1,4]]; np.testing.assert_allclose(actual,recall(jax.tree.map(jnp.asarray,p),perm),atol=2e-6)
 def test_causal_mask(self):
  p=init_model('transformer',2); x=np.arange(10)[None,:]; changed=x.copy(); changed[:,6:]=20
  np.testing.assert_allclose(transformer(p,x)[:,:6],transformer(p,changed)[:,:6],atol=1e-6)
 def test_batch_aggregation_equivalence(self):
  p=init_model('mlp',1); x=jnp.array([1,2,1,4]); y=jnp.array([4,3,3,9]); table=jnp.zeros((512,512)).at[x,y].add(1)
  ordinary=lambda q:optax.softmax_cross_entropy_with_integer_labels(mlp(q,x),y).mean()
  aggregate=lambda q:-jnp.sum(table*jax.nn.log_softmax(mlp(q,jnp.arange(512))))/4
  a,ga=jax.value_and_grad(ordinary)(p); b,gb=jax.value_and_grad(aggregate)(p); self.assertAlmostEqual(float(a),float(b),places=6)
  for u,v in zip(jax.tree.leaves(ga),jax.tree.leaves(gb)): np.testing.assert_allclose(u,v,atol=2e-7)
 def test_sampler_threshold_crossing(self):
  z=np.log([.5,.3,.15,.05]); np.testing.assert_allclose(distribution(z,top_p=.7),[.625,.375,0,0]); np.testing.assert_allclose(distribution(z,top_k=2),[.625,.375,0,0]); np.testing.assert_allclose(distribution(z,min_p=.3),np.array([.5,.3,.15,0])/.95)
  np.testing.assert_allclose(distribution(z,temperature=.5),np.array([.5,.3,.15,.05])**2/np.sum(np.array([.5,.3,.15,.05])**2))
 def test_scaling_gradient_optimum(self):
  t=np.array([np.log(1.8),np.log(500),np.log(2000),.35,.37]); N=np.array([1e8,1e9]); D=np.array([1e10,1e11]); L=np.array([3.,2.5]); val,g=objective(t,np.log(N),np.log(D),np.log(L)); eps=1e-6
  numeric=np.array([(objective(t+np.eye(5)[i]*eps,np.log(N),np.log(D),np.log(L))[0]-objective(t-np.eye(5)[i]*eps,np.log(N),np.log(D),np.log(L))[0])/(2*eps) for i in range(5)]); np.testing.assert_allclose(g,numeric,atol=1e-7)
  p=np.r_[np.exp(t[:3]),t[3:]]; N,D=optimum(p,1e22); self.assertAlmostEqual(6*N*D/1e22,1); self.assertLess(law(p,N,D),law(p,N*1.1,D/1.1)); self.assertLess(law(p,N,D),law(p,N/1.1,D*1.1))
if __name__=='__main__': unittest.main()
