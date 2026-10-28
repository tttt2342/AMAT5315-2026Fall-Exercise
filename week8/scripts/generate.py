from neural import *
from scipy.special import softmax,logsumexp
import sentencepiece as spm

def distribution(logits,temperature=1.,top_k=None,top_p=None,min_p=None):
 p=softmax(np.asarray(logits,dtype=np.float64)/temperature,axis=-1)
 if top_k is not None:
  order=np.argsort(-p,axis=-1); keep=np.zeros_like(p,dtype=bool); np.put_along_axis(keep,order[...,:top_k],True,axis=-1); p=np.where(keep,p,0)
 if top_p is not None:
  order=np.argsort(-p,axis=-1); sorted_p=np.take_along_axis(p,order,axis=-1); keep_sorted=np.cumsum(sorted_p,axis=-1)-sorted_p<top_p; keep=np.zeros_like(p,dtype=bool); np.put_along_axis(keep,order,keep_sorted,axis=-1); p=np.where(keep,p,0)
 if min_p is not None: p=np.where(p>=min_p*p.max(-1,keepdims=True),p,0)
 return p/p.sum(-1,keepdims=True)

@jax.jit
def next_logits(params,buffer,index): return transformer(params,buffer)[:,index]
def generate(params,seed,count,length,eos,options=None,exclude_eos=False):
 rng=np.random.default_rng(seed); tokens=np.full((count,1),eos,dtype=np.int32); surprises=[]; entropies=[]
 for i in range(length):
  context=tokens[:,-64:]; buffer=np.zeros((count,64),dtype=np.int32); buffer[:,:context.shape[1]]=context
  logits=np.asarray(next_logits(params,buffer,context.shape[1]-1)); logp=logits.astype(np.float64)-logsumexp(logits.astype(np.float64),axis=-1,keepdims=True); base=np.exp(logp); p=distribution(logits,**(options or {}))
  if exclude_eos: p[:,eos]=0; p/=p.sum(-1,keepdims=True)
  chosen=np.sum(rng.random(count)[:,None]>np.cumsum(p,axis=-1),axis=-1); chosen=np.minimum(chosen,511).astype(np.int32)
  surprises.append(-logp[np.arange(count),chosen]); entropies.append(-np.sum(base*logp,axis=-1)); tokens=np.concatenate([tokens,chosen[:,None]],axis=1)
 return tokens[:,1:],np.array(surprises).T,np.array(entropies).T

def main():
 p=load('transformer'); data,meta=corpus(); eos=meta['eos']; sp=spm.SentencePieceProcessor(model_file=str(DATA/'tok512.model'))
 settings={'temperature':{'temperature':.8},'top_k':{'temperature':.8,'top_k':20},'top_p':{'temperature':.8,'top_p':.9},'min_p':{'temperature':.7,'min_p':.1}}
 samples={}
 for name,opt in settings.items():
  tokens,_,_=generate(p,1848,1,128,eos,opt); samples[name]={'settings':opt,'text':sp.decode(tokens[0].tolist()),'tokens':tokens[0]}
 tokens,_,_=generate(p,1848,1,60,eos,exclude_eos=True); samples['temperature_1_comparison']={'text':sp.decode(tokens[0].tolist()),'tokens':tokens[0]}
 ctx=sp.encode('The little'); logits=np.asarray(transformer(p,jnp.array([ctx])))[:,-1][0]; tests={}; rng=np.random.default_rng(1855)
 for name,opt in settings.items():
  prob=distribution(logits,**opt); draws=rng.choice(512,100000,p=prob); tests[name]=chi_test(np.bincount(draws,minlength=512),prob)
 greedy=distribution(logits,temperature=.0001); draws=rng.choice(512,100000,p=greedy); tests['greedy_all_argmax']=bool(np.all(draws==np.argmax(logits)))
 consistency={}; raw={}
 for temperature in [1.,.7]:
  tokens,s,h=generate(p,1856,128,128,eos,{'temperature':temperature}); delta=(s-h).mean(1); se=delta.std(ddof=1)/np.sqrt(len(delta)); consistency[str(temperature)]={'mean_surprise':float(s.mean()),'mean_entropy':float(h.mean()),'sequence_standard_error':float(se),'z':float(delta.mean()/se),'sequences':128,'length':128,'seed':1856}; raw[str(temperature)+'_tokens']=tokens; raw[str(temperature)+'_surprise']=s; raw[str(temperature)+'_entropy']=h
 np.savez(ART/'generation.npz',**raw,fixed_logits=logits); save('generation',{'samples':samples,'sampler_tests':tests,'consistency':consistency,'fixed_context':'The little','context_note':'Once upon a time produced single-token top-p and min-p distributions (zero chi-square degrees of freedom); changed to The little before inspecting test p-values.','sample_seed':1848,'chi_seed':1855})
 counts=json.loads((EVI/'counts.json').read_text()); t=json.loads((EVI/'transformer.json').read_text()); ml=json.loads((EVI/'mlp.json').read_text())
 names=['uniform','1','2','3','4','5','6']; values=[counts[k]['bpc'] for k in names]+[ml['bpc'],t['bpc']]; labels=['uniform']+[f'{n}-gram' for n in range(1,7)]+['MLP','Transformer']
 plt.figure(figsize=(10,4)); plt.bar(labels,values,color=['#7c97a5']*7+['#ccaa66','#bb5544']); plt.xticks(rotation=25); plt.ylabel('held-out bits / character'); fig('comparison')
 plt.figure(figsize=(8,4)); pos=np.arange(1,65); loss=np.array(t['position_loss']); se=np.array(t['position_se']); plt.plot(pos,loss,label='Transformer'); plt.fill_between(pos,loss-2*se,loss+2*se,alpha=.2)
 for n in [2,3]: plt.axhline(counts[str(n)]['nats'],ls='--',label=f'{n}-gram')
 plt.xlabel('tokens read'); plt.ylabel('held-out nats / token'); plt.legend(); fig('context')
 print(tests,consistency,flush=True)
if __name__=='__main__': main()
