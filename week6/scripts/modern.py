"""One modern Hopfield update equals explicitly scaled JAX attention."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MPLCONFIGDIR','/tmp/amat5315-mpl')
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from data import ARTIFACTS,EVIDENCE,mnist,show_image
from hopfield import hebbian,recall_batch


def modern_update(patterns,queries,beta):
    weights=jax.nn.softmax(beta*(queries@patterns.T),axis=-1)
    return weights@patterns,weights


def attention_update(patterns,queries,beta):
    # Queries are T x 1 x 784; keys and values are P x 1 x 784.
    return jax.nn.dot_product_attention(queries[:,None,:],patterns[:,None,:],patterns[:,None,:],
                                       scale=beta,implementation='xla')[:,0,:]


def modern_energy(patterns,state,beta):
    return -jax.scipy.special.logsumexp(beta*(state@patterns.T),axis=-1)/beta+.5*jnp.sum(state*state,axis=-1)


def retrieval_success(outputs,patterns,target_ids):
    signs=np.where(np.asarray(outputs)>=0,1,-1).astype(np.float32)
    similarities=signs@np.asarray(patterns).T/signs.shape[1]
    target_overlap=similarities[np.arange(len(signs)),target_ids]
    # A tied nearest target counts as nearest; ties are separately reported.
    nearest=target_overlap>=similarities.max(axis=1)-1e-7
    ties=((similarities==similarities.max(axis=1,keepdims=True)).sum(1)>1)
    return (target_overlap>=.95)&nearest,target_overlap,ties


def main():
    seed=6640; ps=[3,10,30,100,300,1000]; betas=[.001,.003,.01,.03,.1,.3,1.]
    images,labels=mnist('train')
    all_patterns=2*(images[:1000]>=128).reshape(1000,784).astype(np.float32)-1
    # Fixed random lower halves shared by every beta and classical recall.
    noise=np.random.default_rng(seed).choice(np.array([-1,1],np.float32),size=(100,392))
    summary=dict(seed=seed,P=ps,beta=betas,queries=100,models={},max_attention_difference=0.,tie_rule='target may tie for nearest; zero output maps to +1')
    begin=time.perf_counter(); panel=None
    for p in ps:
        patterns=jnp.array(all_patterns[:p]); ids=np.arange(100)%p
        targets=all_patterns[ids]; queries=targets.copy(); queries[:,392:]=noise
        query=jnp.array(queries)
        matrix=hebbian(patterns)
        classical,sweeps=recall_batch(matrix,query.astype(jnp.int32),jax.random.split(jax.random.fold_in(jax.random.key(seed),p),100))
        classical=np.array(classical)
        assert np.all(classical*(classical@np.array(matrix))>=0)
        ok,overlap,ties=retrieval_success(classical,patterns,ids)
        stats=dict(classical=float(ok.mean()),classical_overlap=float(overlap.mean()),modern=[],attention_differences=[],nearest_ties=[],max_sweeps=int(np.max(sweeps)),
                   stored_unstable=float(jnp.mean(patterns*(patterns@matrix.astype(jnp.float32))<0)))
        outputs=[]; all_weights=[]
        for beta in betas:
            output,weights=modern_update(patterns,query,beta)
            attention=attention_update(patterns,query,beta)
            difference=float(jnp.max(jnp.abs(output-attention)))
            assert difference<1e-5,difference
            assert np.all(np.array(modern_energy(patterns,output,beta))<=np.array(modern_energy(patterns,query,beta))+1e-3)
            ok,overlap,ties=retrieval_success(output,patterns,ids)
            stats['modern'].append(float(ok.mean())); stats['attention_differences'].append(difference)
            stats['nearest_ties'].append(int(ties.sum()))
            outputs.append(np.array(output)); all_weights.append(np.array(weights))
            summary['max_attention_difference']=max(summary['max_attention_difference'],difference)
        summary['models'][str(p)]=stats
        np.savez_compressed(ARTIFACTS/f'modern_P{p}.npz',patterns=np.array(patterns),targets=targets,target_ids=ids,queries=queries,classical=classical,modern=outputs,weights=all_weights,beta=betas,sweeps=sweeps,seed=seed)
        if p==100: panel=(targets,queries,classical,np.array(outputs),np.array(all_weights))
        print(f'P={p}: classical={stats["classical"]:.2f}, modern={stats["modern"]}, max difference={max(stats["attention_differences"]):.3g}',flush=True)
    summary['seconds']=time.perf_counter()-begin
    for folder in (ARTIFACTS,EVIDENCE): (folder/'modern.json').write_text(json.dumps(summary,indent=2))
    targets,queries,classical,outputs,weights=panel
    fig,axes=plt.subplots(3,1,figsize=(10,7),layout='constrained')
    for ax,beta in zip(axes,[.003,.03,.3]):
        i=betas.index(beta); ax.bar(np.arange(100),weights[i,0],width=1)
        ax.set(xlabel='Stored pattern index (target = 0)',ylabel='Weight',title=f'beta={beta}; target weight={weights[i,0,0]:.5f}')
    fig.savefig(EVIDENCE/'attention_weights.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(5,10,figsize=(15,8),layout='constrained')
    names=['Target','Query','Classical']+[f'beta={b:g}' for b in betas]
    for row in range(5):
        panels=[targets[row],queries[row],classical[row]]+[out[row] for out in outputs]
        for col,(image,name) in enumerate(zip(panels,names)):
            show_image(axes[row,col],image.reshape(28,28),name if row==0 else None,-1,1)
    fig.suptitle('P=100: first five queries; modern outputs shown before taking sign')
    fig.savefig(EVIDENCE/'modern_beta_gallery.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(6,4,figsize=(7,10),layout='constrained')
    for row in range(6):
        for col,(image,name) in enumerate(zip([targets[row],queries[row],classical[row],outputs[5,row]],['Target','Query','Classical','Modern beta=0.3'])):
            show_image(axes[row,col],image.reshape(28,28),name if row==0 else None,-1,1)
    fig.suptitle('P=100: same stored patterns and same queries')
    fig.savefig(EVIDENCE/'modern_retrieval.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    ax.plot(ps,[summary['models'][str(p)]['classical'] for p in ps],'ko--',label='Classical')
    for bi,beta in enumerate(betas):
        ax.plot(ps,[summary['models'][str(p)]['modern'][bi] for p in ps],'o-',label=f'Modern beta={beta:g}')
    ax.set(xscale='log',xlabel='Stored digits P',ylabel='Retrieval success (100 queries)',ylim=(-.03,1.03))
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(EVIDENCE/'modern_success.png',dpi=160); plt.close(fig)
    assert all(s['modern'][5]>=.90 for s in summary['models'].values())
    assert all(s['classical']==0 for s in summary['models'].values())
    print('Maximum attention difference:',summary['max_attention_difference'],flush=True)

if __name__=='__main__': main()
