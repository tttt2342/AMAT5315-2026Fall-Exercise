"""Binary JAX RBM: matched CD-k training, fresh generation, clamped inference."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/amat5315-mpl')
import argparse
from functools import partial
import hashlib
import json
import time
import tomllib
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from data import ROOT, ARTIFACTS, EVIDENCE, mnist, gallery, show_image


def energy(params, v, h):
    w, a, b = params
    return -v @ a - h @ b - jnp.sum((v @ w) * h, axis=-1)


def free_energy(params, v):
    w, a, b = params
    return -v @ a - jnp.sum(jax.nn.softplus(v @ w + b), axis=-1)


def gibbs_step(params, v, key):
    w, a, b = params
    kh, kv = jax.random.split(key)
    h = jax.random.bernoulli(kh, jax.nn.sigmoid(v @ w + b)).astype(jnp.float32)
    return jax.random.bernoulli(kv, jax.nn.sigmoid(h @ w.T + a)).astype(jnp.float32)


def gibbs(params, v, key, steps):
    return jax.lax.fori_loop(0, steps, lambda i,s: gibbs_step(params,s,jax.random.fold_in(key,i)), v)


def surrogate(params, data, endpoint):
    return jnp.mean(free_energy(params, data) - free_energy(params, jax.lax.stop_gradient(endpoint)))


@partial(jax.jit, static_argnames=('k',))
def epoch_train(params, batches, keys, k, rate):
    def update(params, pair):
        batch, key = pair
        endpoint = gibbs(params, batch, key, k)
        value, gradient = jax.value_and_grad(surrogate)(params, batch, endpoint)
        return jax.tree.map(lambda p,g: p-rate*g, params, gradient), value
    return jax.lax.scan(update, params, (batches, keys))

reconstruct = jax.jit(gibbs_step)


def initialize(train, config):
    c = config['initialization']
    frequency = jnp.clip(train.mean(axis=0), *c['pixel_frequency_clip'])
    return (c['weight_std']*jax.random.normal(jax.random.key(c['seed']), (784,128), dtype=jnp.float32),
            jnp.log(frequency/(1-frequency)), jnp.full(128,c['hidden_bias'],dtype=jnp.float32))


def load_params(k):
    with np.load(ARTIFACTS / f'cd{k}_epoch20.npz') as f:
        return tuple(jnp.array(f[name]) for name in ('w','a','b'))


def train_models(config):
    tr = config['training']
    train = jnp.array((mnist('train')[0]>=128).reshape(tr['images'],784),dtype=jnp.float32)
    test = jnp.array((mnist('test')[0][:config['evaluation']['test_images']]>=128).reshape(-1,784),dtype=jnp.float32)
    init = initialize(train,config)
    eval_key = jax.random.key(config['evaluation']['seed'])
    history = dict(settings=config, hidden_units=128, backend=jax.default_backend(), jax=jax.__version__,
        initialization_sha256=hashlib.sha256(b''.join(np.asarray(x).tobytes() for x in init)).hexdigest(), models={})
    for k in (1,10):
        params=init
        errors=[float(jnp.mean(reconstruct(params,test,eval_key)!=test))]
        values=[]; begin=time.perf_counter(); order_hash=hashlib.sha256()
        for epoch in range(tr['epochs']):
            # Separate shuffle/chain keys; k never changes data order.
            epoch_key=jax.random.fold_in(jax.random.key(tr['seed']),epoch)
            shuffle_key,sample_key=jax.random.split(epoch_key)
            order=jax.random.permutation(shuffle_key,len(train))
            order_hash.update(np.asarray(order).tobytes())
            batches=train[order].reshape(-1,tr['batch_size'],784)
            keys=jax.random.split(sample_key,len(batches))
            params,losses=epoch_train(params,batches,keys,k,tr['learning_rate'])
            values.append(float(jnp.mean(losses)))
            errors.append(float(jnp.mean(reconstruct(params,test,eval_key)!=test)))
            np.savez_compressed(ARTIFACTS/f'cd{k}_epoch{epoch+1}.npz',w=params[0],a=params[1],b=params[2])
            print(f'CD-{k} epoch {epoch+1:02d}: surrogate={values[-1]:.4f}, Hamming={errors[-1]:.5f}',flush=True)
        history['models'][str(k)]=dict(surrogate=values,reconstruction=errors,seconds=time.perf_counter()-begin,shuffle_sha256=order_hash.hexdigest())
        (ARTIFACTS/'rbm_training.json').write_text(json.dumps(history,indent=2))
    assert history['models']['1']['shuffle_sha256']==history['models']['10']['shuffle_sha256']
    for model in history['models'].values():
        assert model['reconstruction'][-1]<.5*model['reconstruction'][0]
    (EVIDENCE/'rbm_training.json').write_text(json.dumps(history,indent=2))


def training_figures():
    hist=json.loads((ARTIFACTS/'rbm_training.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for k,model in hist['models'].items():
        axes[0].plot(range(1,21),model['surrogate'],label=f'CD-{k}')
        axes[1].plot(range(21),model['reconstruction'],label=f'CD-{k}')
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=.2)
    axes[0].set_ylabel('CD surrogate (pre-update batch mean; not NLL)')
    axes[1].set_ylabel('Held-out one-step Hamming fraction')
    fig.savefig(EVIDENCE/'rbm_training.png',dpi=160); plt.close(fig)
    images,labels=mnist('train')
    gallery(2*(images[:16]>=128).astype(np.int8)-1,labels[:16],columns=8,filename='binary_spins.png',
            title='First 16 training images: ink +1, background -1',vmin=-1,vmax=1)
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    show_image(axes[0],2*(images[0]>=128).astype(np.int8)-1,'One 28 x 28 spin configuration',-1,1)
    ax=axes[1]
    for i in range(8):
        for j in range(5): ax.plot([i,j*1.75],[0,1],color='steelblue',alpha=.18)
    ax.scatter(np.arange(8),np.zeros(8),s=130,zorder=3)
    ax.scatter(np.arange(5)*1.75,np.ones(5),s=130,zorder=3)
    ax.text(3.5,-.25,'784 visible bits v; bias a',ha='center')
    ax.text(3.5,1.2,'128 hidden bits h; bias b',ha='center')
    ax.text(3.5,.5,'W: between-layer couplings only',ha='center',bbox={'facecolor':'white','alpha':.9})
    ax.set_ylim(-.5,1.5); ax.axis('off')
    fig.savefig(EVIDENCE/'rbm_structure.png',dpi=160); plt.close(fig)


@partial(jax.jit,static_argnames=('steps','clamp'))
def advance(params,state,key,start,steps,observed,clamp):
    def body(i,v):
        v=gibbs_step(params,v,jax.random.fold_in(key,start+i))
        if clamp: v=jnp.where(jnp.arange(784)[None,:]<392,observed,v)
        return v
    return jax.lax.fori_loop(0,steps,body,state)


def evaluate(config):
    test_images,labels=mnist('test')
    targets=(test_images[:64]>=128).reshape(64,784).astype(np.float32)
    steps=config['completion']['steps']; generation_seed=6605
    start=jax.random.bernoulli(jax.random.key(generation_seed),.5,(64,784)).astype(jnp.float32)
    wrong_indices=np.array([next((i+d)%64 for d in range(1,64) if labels[(i+d)%64]!=labels[i]) for i in range(64)])
    comp_key,noise_key=jax.random.split(jax.random.key(config['completion']['seed']))
    lower_noise=jax.random.bernoulli(noise_key,.5,(64*8,784)).astype(jnp.float32)
    observed=jnp.concatenate([jnp.repeat(jnp.array(targets),8,axis=0),jnp.repeat(jnp.array(targets[wrong_indices]),8,axis=0)])
    comp_start=jnp.concatenate([lower_noise,lower_noise])
    comp_start=jnp.where(jnp.arange(784)[None,:]<392,observed,comp_start)
    summary=dict(generation_seed=generation_seed,completion=config['completion'],wrong_indices=wrong_indices.tolist(),models={})
    begin=time.perf_counter()
    for k in (1,10):
        params=load_params(k); generations=[np.array(start)]; v=start; previous=0
        for step in steps:
            v=advance(params,v,jax.random.key(generation_seed+100),previous,step-previous,start,False)
            generations.append(np.array(v)); previous=step
        reconstructed=np.array(reconstruct(params,jnp.array(targets),jax.random.key(config['evaluation']['seed'])))
        np.savez_compressed(ARTIFACTS/f'cd{k}_generation.npz',steps=[0]+steps,frames=generations,targets=targets,reconstruction=reconstructed)
        fig,axes=plt.subplots(6,7,figsize=(10,9),layout='constrained')
        for row in range(6):
            panels=[targets[row],reconstructed[row]]+[frame[row] for frame in generations]
            names=['Held-out','1-step recon.','Random start']+[f'Gibbs {s}' for s in steps]
            for col,(panel,name) in enumerate(zip(panels,names)):
                show_image(axes[row,col],panel.reshape(28,28),name if row==0 else None,0,1)
        fig.suptitle(f'CD-{k}: first six chains, no sample selection')
        fig.savefig(EVIDENCE/f'cd{k}_generation.png',dpi=160); plt.close(fig)
        v=comp_start; frames=[np.array(v)]; previous=0
        for step in steps:
            v=advance(params,v,comp_key,previous,step-previous,observed,True)
            frame=np.array(v)
            assert np.array_equal(frame[:,:392],np.array(observed[:,:392]))
            frames.append(frame); previous=step
            print(f'CD-{k} completion step {step}',flush=True)
        frames=np.array(frames).reshape(5,2,64,8,784)
        lower=frames[1:,:,:,:,392:]; truth=targets[None,None,:,None,392:]
        denominator=lower.sum(-1)+truth.sum(-1)
        dice=np.divide(2*(lower*truth).sum(-1),denominator,out=np.zeros_like(denominator),where=denominator>0)
        digit_mean=dice.mean(axis=-1)
        mean=digit_mean.mean(axis=-1); se=digit_mean.std(axis=-1,ddof=1)/np.sqrt(64)
        summary['models'][str(k)]=dict(true_mean=mean[:,0].tolist(),wrong_mean=mean[:,1].tolist(),
            true_se=se[:,0].tolist(),wrong_se=se[:,1].tolist(),final_advantage=float(mean[-1,0]-mean[-1,1]))
        np.savez_compressed(ARTIFACTS/f'cd{k}_completion.npz',frames=frames,dice=dice,targets=targets,wrong_indices=wrong_indices)
        fig,axes=plt.subplots(8,6,figsize=(9,12),layout='constrained')
        for row in range(8):
            digit,chain=divmod(row,2)
            panels=[targets[digit]]+[frame[0,digit,chain] for frame in frames]
            names=[f'Target {labels[digit]}',f'Start chain {chain+1}']+[f'Step {s}' for s in steps]
            for col,(panel,name) in enumerate(zip(panels,names)):
                show_image(axes[row,col],panel.reshape(28,28),name,0,1)
                if col: axes[row,col].axhline(13.5,color='tomato',lw=.6)
        fig.suptitle(f'CD-{k}: clamped upper half; first four digits, first two chains')
        fig.savefig(EVIDENCE/f'cd{k}_completion.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for k,model in summary['models'].items():
        for condition,style in [('true','-'),('wrong','--')]:
            ax.errorbar(steps,model[f'{condition}_mean'],yerr=model[f'{condition}_se'],fmt='o'+style,capsize=3,label=f'CD-{k}, {condition} upper half')
    ax.set(xscale='log',xlabel='Gibbs steps',ylabel='Lower-half ink overlap (Dice)',ylim=(0,.7))
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(EVIDENCE/'completion_overlap.png',dpi=160); plt.close(fig)
    summary['seconds']=time.perf_counter()-begin
    for folder in (ARTIFACTS,EVIDENCE): (folder/'rbm_evaluation.json').write_text(json.dumps(summary,indent=2))
    for model in summary['models'].values(): assert model['final_advantage']>=.1,model
    print(json.dumps(summary['models'],indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['train','evaluate','all'],default='all',nargs='?')
    args=parser.parse_args(); config=tomllib.loads((ROOT/'rbm.toml').read_text())
    if args.stage in ('train','all'): train_models(config)
    training_figures()
    if args.stage in ('evaluate','all'): evaluate(config)
