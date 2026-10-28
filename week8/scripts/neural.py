"""JAX models, CPU; each linear map includes bias."""
from common import *
import jax
import jax.numpy as jnp
import optax
import pickle,time,sys

def dense_init(r,a,b,s=None): return {'w':jnp.array(r.normal(0,s or a**-.5,(a,b)),dtype=jnp.float32),'b':jnp.zeros(b)}
def linear(p,x): return x@p['w']+p['b']
def norm_init(d): return {'g':jnp.ones(d),'b':jnp.zeros(d)}
def norm(p,x): return (x-x.mean(-1,keepdims=True))*jax.lax.rsqrt(x.var(-1,keepdims=True)+1e-5)*p['g']+p['b']
def init_model(kind,seed):
 r=np.random.default_rng(seed)
 if kind=='mlp': return {'e':jnp.array(r.normal(0,.1,(512,64)),dtype=jnp.float32),'h':dense_init(r,64,512),'out':dense_init(r,512,512)}
 if kind=='recall': return {'e':jnp.array(r.normal(0,.2,(72,64)),dtype=jnp.float32),**{k:dense_init(r,64,64) for k in ['q','k','v','o']},'ln':norm_init(64),'out':dense_init(r,64,8)}
 d=96
 return {'e':jnp.array(r.normal(0,.02,(512,d)),dtype=jnp.float32),'pos':jnp.array(r.normal(0,.02,(64,d)),dtype=jnp.float32),'blocks':[{'ln1':norm_init(d),'ln2':norm_init(d),'qkv':dense_init(r,d,3*d,.02),'o':dense_init(r,d,d,.01),'h':dense_init(r,d,4*d,.02),'proj':dense_init(r,4*d,d,.01)} for _ in range(2)],'ln':norm_init(d),'out':dense_init(r,d,512,.02)}
def mlp(p,x): return linear(p['out'],jnp.tanh(linear(p['h'],p['e'][x])))
def recall(p,x,weights=False):
 h=p['e'][x]; q,k,v=[linear(p[z],h) for z in ['q','k','v']]; a=jax.nn.softmax(q@jnp.swapaxes(k,-1,-2)/8,axis=-1)
 out=linear(p['out'],norm(p['ln'],h+linear(p['o'],a@v)))
 return (out[:,-1],a) if weights else out[:,-1]
def transformer(p,x):
 h=p['e'][x]+p['pos'][:x.shape[1]]; B,T,D=h.shape
 for b in p['blocks']:
  y=linear(b['qkv'],norm(b['ln1'],h)).reshape(B,T,3,4,24); q,k,v=[y[:,:,i].transpose(0,2,1,3) for i in range(3)]
  scores=q@k.swapaxes(-1,-2)/np.sqrt(24); a=jax.nn.softmax(jnp.where(jnp.arange(T)[:,None]>=jnp.arange(T)[None,:],scores,-jnp.inf),axis=-1)
  h=h+linear(b['o'],(a@v).transpose(0,2,1,3).reshape(B,T,D)); h=h+linear(b['proj'],jax.nn.gelu(linear(b['h'],norm(b['ln2'],h))))
 return linear(p['out'],norm(p['ln'],h))
def recall_batch(r,batch):
 keys=np.argsort(r.random((batch,8)),axis=1)[:,:4]; vals=r.integers(0,8,(batch,4)); idx=r.integers(0,4,batch)
 return np.concatenate([keys*8+vals,(64+keys[np.arange(batch),idx])[:,None]],axis=1).astype(np.int32),vals[np.arange(batch),idx].astype(np.int32),4-idx

def train(kind):
 seed={'mlp':1803,'recall':1804,'transformer':1805}[kind]; r=np.random.default_rng(seed); p=init_model(kind,seed)
 steps=6000 if kind=='transformer' else 4000; peak=.006 if kind=='mlp' else .003; warm=300 if kind=='transformer' else 200
 schedule=optax.warmup_cosine_decay_schedule(0,peak,warm,steps,end_value=peak*.1); tx=optax.adamw(schedule,weight_decay=.01); state=tx.init(p)
 fn={'mlp':mlp,'recall':recall,'transformer':transformer}[kind]
 def loss(p,x,y):
  # Aggregate duplicated previous-token inputs, exactly preserving the sampled batch gradient.
  if kind=='mlp': return -jnp.sum(y*jax.nn.log_softmax(mlp(p,jnp.arange(512)),axis=-1))/16384
  return optax.softmax_cross_entropy_with_integer_labels(fn(p,x),y).mean()
 @jax.jit
 def step(p,s,x,y):
  l,g=jax.value_and_grad(loss)(p,x,y); updates,s=tx.update(g,s,p); return optax.apply_updates(p,updates),s,l
 if kind!='recall': data,meta=corpus(); tokens=data['train']; prev=np.concatenate([[meta['eos']],tokens[:-1]])
 history=[]; start=time.monotonic()
 for i in range(steps):
  if kind=='mlp':
   idx=r.integers(0,len(tokens),16384); x=np.zeros(1,dtype=np.int32); y=np.bincount(prev[idx]*512+tokens[idx],minlength=512*512).reshape(512,512).astype(np.float32)
  elif kind=='recall': x,y,_=recall_batch(r,256)
  else:
   idx=r.integers(0,len(tokens)-64,32); batch=tokens[idx[:,None]+np.arange(65)]; x,y=batch[:,:-1],batch[:,1:]
  p,state,l=step(p,state,x,y)
  if i%100==0 or i==steps-1: history.append([i+1,float(l)]); print(kind,i+1,float(l),round(time.monotonic()-start,1),flush=True)
 jax.block_until_ready(p); elapsed=time.monotonic()-start
 with open(ART/f'{kind}.pkl','wb') as f: pickle.dump(jax.tree.map(np.asarray,p),f)
 settings={'seed':seed,'steps':steps,'batch':{'mlp':16384,'recall':256,'transformer':32}[kind],'peak_lr':peak,'warmup':warm,'end_lr':peak*.1,'weight_decay':.01,'backend':jax.default_backend(),'seconds':elapsed,'parameters':sum(a.size for a in jax.tree.leaves(p)),'history':history}
 if kind=='mlp':
  logp=np.asarray(jax.nn.log_softmax(mlp(p,jnp.arange(512)),axis=-1)); val=data['val']; before=np.concatenate([[meta['eos']],val[:-1]])
  settings['nats']=float(-logp[before,val].mean()); settings['bpc']=settings['nats']*len(val)/np.log(2)/meta['val']['characters']; np.save(ART/'mlp_logp.npy',logp)
 elif kind=='recall':
  x,y,dist=recall_batch(np.random.default_rng(1904),20000); logits,a=jax.jit(lambda p,x:recall(p,x,True))(p,x); correct=np.asarray(logits).argmax(-1)==y
  settings['accuracy_by_distance']={str(d):float(correct[dist==d].mean()) for d in range(1,5)}; settings['test_sequences']=20000; settings['example_tokens']=x[0]; settings['example_target']=int(y[0]); settings['example_weights']=np.asarray(a[0])
  plt.figure(figsize=(9,4)); ax=plt.subplot(121); ax.plot(range(1,5),list(settings['accuracy_by_distance'].values()),'o-'); ax.axhline(.95,ls=':',color='gray'); ax.set(xlabel='query-memory distance',ylabel='accuracy',ylim=(0,1.05)); ax=plt.subplot(122); im=ax.imshow(a[0],vmin=0,vmax=1); plt.colorbar(im,ax=ax); ax.set(xlabel='read position',ylabel='output position'); fig('recall')
 else:
  val=data['val']; inp=np.concatenate([[meta['eos']],val]); losses=[]; evaluate=jax.jit(lambda p,x,y:optax.softmax_cross_entropy_with_integer_labels(transformer(p,x),y))
  for begin in range(0,len(val),4096):
   starts=np.arange(begin,min(begin+4096,len(val)),64); x=inp[np.minimum(starts[:,None]+np.arange(64),len(inp)-1)]; y=inp[np.minimum(starts[:,None]+np.arange(1,65),len(inp)-1)]; losses.append(np.asarray(evaluate(p,x,y)))
  matrix=np.concatenate(losses); flat=matrix.ravel()[:len(val)]; settings['nats']=float(flat.mean()); settings['bpc']=float(flat.sum()/np.log(2)/meta['val']['characters']); settings['position_loss']=matrix[:-1].mean(0); settings['position_se']=matrix[:-1].std(0,ddof=1)/np.sqrt(len(matrix)-1); np.save(ART/'transformer_validation.npy',flat)
 save(kind,settings); print(settings,flush=True)
 plt.figure(figsize=(6,4)); h=np.array(history); plt.plot(h[:,0],h[:,1]); plt.xlabel('step'); plt.ylabel('training cross-entropy'); fig(kind+'_training')
def load(kind):
 with open(ART/f'{kind}.pkl','rb') as f: return jax.tree.map(jnp.asarray,pickle.load(f))
if __name__=='__main__': train(sys.argv[1])
