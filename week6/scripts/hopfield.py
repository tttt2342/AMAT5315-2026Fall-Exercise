"""JAX asynchronous Hopfield recall with exact integer fields (N*J)."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MPLCONFIGDIR','/tmp/amat5315-mpl')
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ndtr
from data import ARTIFACTS,EVIDENCE


def hebbian(patterns):
    """Return N*J as integers; positive scaling preserves all zero-T decisions."""
    x=jnp.asarray(patterns,dtype=jnp.float32)
    matrix=(x.T@x).astype(jnp.int32)
    return matrix.at[jnp.diag_indices(x.shape[1])].set(0)


def energy(matrix,state):
    return -.5*jnp.dot(state.astype(jnp.float32),matrix.astype(jnp.float32)@state)/len(state)


def recall_one(matrix,initial,key):
    initial=initial.astype(jnp.int32)
    field=matrix@initial
    def sweep(carry):
        state,field,key,count,_=carry
        key,order_key=jax.random.split(key)
        order=jax.random.permutation(order_key,len(state))
        def visit(carry,i):
            state,field,flips=carry
            new=jnp.where(field[i]>0,1,jnp.where(field[i]<0,-1,state[i]))
            delta=new-state[i]
            return (state.at[i].set(new),field+delta*matrix[:,i],flips+(delta!=0)),None
        (state,field,flips),_=jax.lax.scan(visit,(state,field,jnp.int32(0)),order)
        return state,field,key,count+1,flips
    # Stop only after an entire sweep flips nothing; no arbitrary truncation.
    state,field,_,sweeps,_=jax.lax.while_loop(lambda c:c[-1]>0,sweep,(initial,field,key,jnp.int32(0),jnp.int32(1)))
    return state,sweeps

recall_batch=jax.jit(jax.vmap(recall_one,in_axes=(None,0,0)))
recall_sets=jax.jit(jax.vmap(jax.vmap(recall_one,in_axes=(None,0,0)),in_axes=(0,0,0)))


def crossing(loads,means,threshold=.9):
    below=np.flatnonzero(means<threshold)
    if len(below)==0: return None
    i=below[0]
    if i==0: return float(loads[0])
    return float(loads[i-1]+(threshold-means[i-1])*(loads[i]-loads[i-1])/(means[i]-means[i-1]))


def main():
    seed=6630; sizes=[100,200,400,800]; loads=np.arange(5,26)/100
    summary=dict(seed=seed,sets_per_point=20,loads=loads.tolist(),sizes=sizes,backend=jax.default_backend(),models={})
    total_start=time.perf_counter()
    for n in sizes:
        start=time.perf_counter(); unstable=[]; overlaps=[]; success=[]; sweep_counts=[]
        for ai,alpha in enumerate(loads):
            p=round(float(alpha)*n)
            patterns=[]; keys=[]
            for rep in range(20):
                # SeedSequence coordinates make every point/set independently reproducible.
                rng=np.random.default_rng(np.random.SeedSequence([seed,n,ai,rep]))
                patterns.append((2*rng.integers(0,2,size=(p,n))-1).astype(np.float32))
                key=jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(jax.random.key(seed),n),ai),rep)
                keys.append(jax.random.split(key,min(p,10)))
            patterns=np.array(patterns)
            matrices=jax.vmap(hebbian)(jnp.array(patterns))
            fields=jnp.einsum('bpi,bij->bpj',jnp.array(patterns),matrices.astype(jnp.float32))
            fractions=np.array(jnp.mean(jnp.array(patterns)*fields<0,axis=(1,2)))
            initial=jnp.array(patterns[:,:min(p,10)],dtype=jnp.int32)
            final,sweeps=recall_sets(matrices,initial,jnp.stack(keys))
            final=np.array(final); sweeps=np.array(sweeps)
            ov=np.mean(final*np.array(initial),axis=-1)
            assert np.all(final*np.einsum('bij,bqj->bqi',np.array(matrices),final)>=0)
            unstable.append(fractions); overlaps.append(ov.mean(1)); success.append((ov>=.9).mean(1)); sweep_counts.append(sweeps)
            np.savez_compressed(ARTIFACTS/f'hopfield_N{n}_P{p}.npz',patterns=patterns.astype(np.int8),final=final.astype(np.int8),overlap=ov,unstable=fractions,sweeps=sweeps,seed=seed,seed_coordinates=[n,ai])
            print(f'N={n} P={p}: unstable={fractions.mean():.5f}, overlap={ov.mean():.4f}, success={(ov>=.9).mean():.3f}, max sweeps={sweeps.max()}',flush=True)
        stats={}
        for name,values in [('unstable',unstable),('overlap',overlaps),('success',success)]:
            values=np.array(values)
            stats[name]=values.mean(1).tolist(); stats[name+'_se']=(values.std(1,ddof=1)/np.sqrt(20)).tolist()
        stats['crossing']=crossing(loads,np.array(stats['overlap']))
        stats['seconds']=time.perf_counter()-start
        stats['max_sweeps']=int(max(x.max() for x in sweep_counts))
        summary['models'][str(n)]=stats
        (ARTIFACTS/'hopfield.json').write_text(json.dumps(summary,indent=2))
    summary['seconds']=time.perf_counter()-total_start
    for folder in (ARTIFACTS,EVIDENCE): (folder/'hopfield.json').write_text(json.dumps(summary,indent=2))
    plot(summary)
    crossings=[summary['models'][str(n)]['crossing'] for n in sizes]
    assert all(a>b for a,b in zip(crossings,crossings[1:])),crossings
    assert np.all(np.abs(np.array(crossings)-[.186,.170,.156,.150])<=.02),crossings
    print('Capacity crossings:',crossings,flush=True)


def plot(summary):
    loads=np.array(summary['loads'])
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for n,stats in summary['models'].items():
        line=axes[0].errorbar(loads,stats['unstable'],yerr=stats['unstable_se'],fmt='o-',ms=3,capsize=2,label=f'N={n}')
        finite=ndtr(-np.sqrt((int(n)-1)/(np.rint(loads*int(n))-1)))
        axes[1].plot(loads,finite,'--',color=line[0].get_color(),label=f'Gaussian finite N={n}')
        axes[1].errorbar(loads,stats['unstable'],yerr=stats['unstable_se'],fmt='o',ms=3,color=line[0].get_color())
    for ax in axes:
        ax.plot(loads,ndtr(-1/np.sqrt(loads)),'k-',label='Gaussian large N')
        ax.set(xlabel='Load alpha = P/N',ylabel='Unstable-bit fraction'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(EVIDENCE/'hopfield_crosstalk.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for n,stats in summary['models'].items():
        for ax,name in zip(axes,['overlap','success']):
            ax.errorbar(loads,stats[name],yerr=stats[name+'_se'],fmt='o-',ms=3,capsize=2,label=f'N={n}')
    for ax in axes:
        ax.axvline(.138,color='k',ls='--',label='alpha_c = 0.138')
        ax.set(xlabel='Load alpha = P/N',ylim=(0,1.03)); ax.legend(fontsize=8); ax.grid(alpha=.2)
    axes[0].axhline(.9,color='gray',ls=':'); axes[0].set_ylabel('Mean final target overlap')
    axes[1].set_ylabel('Fraction with overlap >= 0.9')
    fig.savefig(EVIDENCE/'hopfield_capacity.png',dpi=160); plt.close(fig)

if __name__=='__main__': main()
