from common import *
from scipy.stats import chi2
import sentencepiece as spm
V=512

def codes(tokens,n,eos=2):
    # Exactly one prediction per token, initial history consists of EOS tokens.
    padded=np.pad(np.asarray(tokens,dtype=np.uint64),(n-1,0),constant_values=eos)
    out=np.zeros(len(tokens),dtype=np.uint64)
    for i in range(n): out=(out<<np.uint64(9))|padded[i:i+len(tokens)]
    return out

def lookup(keys,values,q):
    pos=np.searchsorted(keys,q); safe=np.minimum(pos,len(keys)-1)
    return np.where((pos<len(keys))&(keys[safe]==q),values[safe],0)

def fit(tokens,n,eos):
    keys,vals=np.unique(codes(tokens,n,eos),return_counts=True)
    contexts,idx=np.unique(keys>>np.uint64(9),return_index=True)
    totals=np.add.reduceat(vals,idx)
    return keys,vals,contexts,totals

def score(model,tokens,n,k,eos):
    keys,vals,ctx,total=model; q=codes(tokens,n,eos)
    counts=lookup(keys,vals,q); den=lookup(ctx,total,q>>np.uint64(9))
    return float(np.mean(np.log(den+k*V)-np.log(counts+k))),float(np.mean(counts==0))

def main():
    data,meta=corpus(); train=data['train']; val=data['val']; eos=meta['eos']; split=int(.9*len(train)); results={}
    for n in range(1,7):
        tuning=fit(train[:split],n,eos); grid=[.0001,.001,.01,.1,1]
        losses=[score(tuning,train[split:],n,k,eos)[0] for k in grid]; k=grid[int(np.argmin(losses))]; del tuning
        model=fit(train,n,eos); loss,unseen=score(model,val,n,k,eos)
        results[str(n)]={'k':k,'tuning_losses':losses,'nats':loss,'ppl':np.exp(loss),'bpc':loss*len(val)/np.log(2)/meta['val']['characters'],'unseen':unseen}
        np.savez(ART/f'counts{n}.npz',keys=model[0],counts=model[1],contexts=model[2],totals=model[3],k=k)
        print(n,results[str(n)],flush=True)
    results['uniform']={'nats':np.log(V),'ppl':V,'bpc':np.log2(V)*len(val)/meta['val']['characters']}
    big=np.load(ART/'counts2.npz'); mat=np.zeros((V,V)); mat.flat[big['keys'].astype(int)]=big['counts']; p=(mat+big['k'])/(mat.sum(1,keepdims=True)+V*big['k'])
    # Explicit enumeration, including all 512^3 paths; no matrix-power shortcut.
    first=p[eos]; sums=[float(first.sum()),float((first[:,None]*p).sum())]
    sums.append(float(sum(np.sum(first[a]*p[a,:,None]*p) for a in range(V))))
    rng=np.random.default_rng(1802); seq=np.empty(250001,dtype=int); seq[0]=eos; cdf=p.cumsum(1)
    for i,u in enumerate(rng.random(250000)): seq[i+1]=np.searchsorted(cdf[seq[i]],u)
    sampled=np.bincount(seq[:-1]*V+seq[1:],minlength=V*V).reshape(V,V)
    contexts=np.argsort(np.bincount(train,minlength=V))[-20:]; stats=[chi_stat(sampled[c],p[c]) for c in contexts]; s=sum(x[0] for x in stats); df=sum(x[1] for x in stats)
    results['verification']={'sequence_sums':sums,'max_normalization_error':max(abs(np.array(sums)-1)),'bigram_chi2':{'statistic':s,'df':df,'p':chi2.sf(s,df)},'sample_tokens':250000,'seed':1802}
    np.savez(ART/'bigram.npz',p=p,sampled=sampled)
    sp=spm.SentencePieceProcessor(model_file=str(DATA/'tok512.model')); samples={}
    for n in [0,1,2,3]:
        rng=np.random.default_rng(1848); history=[eos]*max(1,n-1); generated=[]
        if n: m=np.load(ART/f'counts{n}.npz')
        for _ in range(60):
            if n==0: prob=np.ones(V)/V
            else:
                ctx=0
                for t in history[-(n-1):] if n>1 else []: ctx=ctx*V+t
                row=lookup(m['keys'],m['counts'],np.arange(V,dtype=np.uint64)+np.uint64(ctx*V)); prob=row+m['k']; prob=prob/prob.sum()
            # Match reference story openings: EOS is excluded only for displayed samples.
            prob[eos]=0; prob/=prob.sum(); t=int(rng.choice(V,p=prob)); generated.append(t); history.append(t)
        samples[str(n)]=sp.decode(generated)
    save('samples_counts',samples); save('counts',results)
    plt.figure(figsize=(9,4)); ax=plt.subplot(121); ax.plot(range(1,7),[results[str(n)]['bpc'] for n in range(1,7)],'o-'); ax.set(xlabel='n',ylabel='held-out bits / character'); ax=plt.subplot(122); ax.plot(range(1,7),[100*results[str(n)]['unseen'] for n in range(1,7)],'o-'); ax.set(xlabel='n',ylabel='unseen n-grams (%)'); fig('counts')
if __name__=='__main__': main()
