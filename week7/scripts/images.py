"""JAX implementation of pinned Diffusers UNet2DModel and image samplers."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json, time, argparse, importlib.util
import numpy as np
import jax
import jax.numpy as jnp
from safetensors.numpy import load_file
from mixture import ROOT,A,E,AB,BETA,savefig
import matplotlib.pyplot as plt
MODEL=ROOT/'inputs/cifar-model'
REVISION='267b167dc01f0e4e61923ea244e8b988f84deb80'

def load_model():
    config=json.loads((MODEL/'config.json').read_text()); mapped={}
    for name,value in load_file(MODEL/'diffusion_pytorch_model.safetensors').items():
        if '.attentions.' in name:
            for old,new in [('query','to_q'),('key','to_k'),('value','to_v'),('proj_attn','to_out.0')]:
                name=name.replace('.'+old+'.','.'+new+'.')
        if value.ndim==4:value=value.transpose(2,3,1,0)
        mapped[name]=jnp.asarray(value,dtype=jnp.float32)
    return config,mapped

def predictor(config):
    groups=config['norm_num_groups'];eps=config['norm_eps']
    def forward(w,input_nchw,t):
        def linear(x,name):return x@w[name+'.weight'].T+w[name+'.bias']
        def conv(x,name,stride=1,padding=((1,1),(1,1))):
            kernel=w[name+'.weight']
            if kernel.shape[:2]==(1,1):padding=((0,0),(0,0))
            return jax.lax.conv_general_dilated(x,kernel,(stride,stride),padding,dimension_numbers=('NHWC','HWIO','NHWC'))+w[name+'.bias']
        def norm(x,name):
            b,h,ww,c=x.shape;z=x.reshape(b,h,ww,groups,c//groups)
            mean=jnp.mean(z,axis=(1,2,4),keepdims=True);var=jnp.mean((z-mean)**2,axis=(1,2,4),keepdims=True)
            return ((z-mean)*jax.lax.rsqrt(var+eps)).reshape(x.shape)*w[name+'.weight']+w[name+'.bias']
        def resnet(x,emb,name):
            h=conv(jax.nn.silu(norm(x,name+'.norm1')),name+'.conv1')
            h=h+linear(jax.nn.silu(emb),name+'.time_emb_proj')[:,None,None,:]
            h=conv(jax.nn.silu(norm(h,name+'.norm2')),name+'.conv2')
            if name+'.conv_shortcut.weight' in w:x=conv(x,name+'.conv_shortcut')
            return x+h
        def attention(x,name):
            b,h,ww,c=x.shape;z=norm(x,name+'.group_norm').reshape(b,h*ww,c)
            q,k,v=[linear(z,name+'.'+n) for n in ['to_q','to_k','to_v']]
            dim=config['attention_head_dim'] or c;heads=c//dim
            q,k,v=[u.reshape(b,h*ww,heads,dim).transpose(0,2,1,3) for u in [q,k,v]]
            scores=jnp.einsum('bhid,bhjd->bhij',q,k)/jnp.sqrt(jnp.float32(dim))
            out=jnp.einsum('bhij,bhjd->bhid',jax.nn.softmax(scores,axis=-1),v).transpose(0,2,1,3).reshape(b,h*ww,c)
            return x+linear(out,name+'.to_out.0').reshape(x.shape)
        half=config['block_out_channels'][0]//2
        t=jnp.broadcast_to(jnp.asarray(t,dtype=jnp.float32),(input_nchw.shape[0],))
        freq=jnp.exp(-jnp.log(10000.)*jnp.arange(half,dtype=jnp.float32)/(half-config['freq_shift']))
        phase=t[:,None]*freq;pieces=[jnp.sin(phase),jnp.cos(phase)]
        if config['flip_sin_to_cos']:pieces.reverse()
        emb=linear(jax.nn.silu(linear(jnp.concatenate(pieces,axis=-1),'time_embedding.linear_1')),'time_embedding.linear_2')
        x=conv(input_nchw.transpose(0,2,3,1),'conv_in');skips=[x]
        for i,kind in enumerate(config['down_block_types']):
            for j in range(config['layers_per_block']):
                x=resnet(x,emb,f'down_blocks.{i}.resnets.{j}')
                if kind=='AttnDownBlock2D':x=attention(x,f'down_blocks.{i}.attentions.{j}')
                skips.append(x)
            if i<len(config['down_block_types'])-1:
                if config['downsample_padding']==0:
                    x=jnp.pad(x,((0,0),(0,1),(0,1),(0,0)));padding=((0,0),(0,0))
                else:padding=((1,1),(1,1))
                x=conv(x,f'down_blocks.{i}.downsamplers.0.conv',2,padding);skips.append(x)
        x=resnet(x,emb,'mid_block.resnets.0');x=attention(x,'mid_block.attentions.0');x=resnet(x,emb,'mid_block.resnets.1')
        for i,kind in enumerate(config['up_block_types']):
            for j in range(config['layers_per_block']+1):
                x=resnet(jnp.concatenate([x,skips.pop()],axis=-1),emb,f'up_blocks.{i}.resnets.{j}')
                if kind=='AttnUpBlock2D':x=attention(x,f'up_blocks.{i}.attentions.{j}')
            if i<len(config['up_block_types'])-1:
                x=jnp.repeat(jnp.repeat(x,2,axis=1),2,axis=2);x=conv(x,f'up_blocks.{i}.upsamplers.0.conv')
        assert not skips
        return conv(jax.nn.silu(norm(x,'conv_norm_out')),'conv_out').transpose(0,3,1,2)
    return jax.jit(forward)

def verify():
    import torch
    from diffusers import UNet2DModel
    torch.set_num_threads(4);config,w=load_model();fn=predictor(config)
    model=UNet2DModel.from_pretrained(str(MODEL),local_files_only=True).eval()
    rng=np.random.default_rng(7240);checks=[]
    for t in [0,49,499,999]:
        x=rng.normal(size=(1,3,32,32)).astype(np.float32)
        with torch.no_grad():ref=model(torch.from_numpy(x),t).sample.numpy()
        got=np.asarray(fn(w,jnp.array(x),t));checks.append(dict(model_index=t,max_abs=float(abs(ref-got).max()),relative_rms=float(np.linalg.norm(ref-got)/np.linalg.norm(ref))))
        print(checks[-1],flush=True);np.testing.assert_allclose(got,ref,atol=3e-5,rtol=3e-4)
    (E/'unet_verification.json').write_text(json.dumps(dict(reference='diffusers==0.40.0 CPU float32',revision=REVISION,checks=checks),indent=2))

def display(x):return np.clip((np.asarray(x).transpose(0,2,3,1)+1)/2,0,1)

def generate():
    config,w=load_model();fn=predictor(config);key=jax.random.PRNGKey(7200)
    x=jax.random.normal(key,(4,3,32,32),dtype=jnp.float32);stages={1000:np.asarray(x)};begin=time.perf_counter()
    @jax.jit
    def step(w,x,t):
        noise=jax.random.normal(jax.random.fold_in(key,t),x.shape);beta=BETA[t-1]
        return (x-beta/jnp.sqrt(1-AB[t])*fn(w,x,t-1))/jnp.sqrt(1-beta)+jnp.where(t>1,jnp.sqrt(beta),0)*noise
    for t in range(1000,0,-1):
        x=step(w,x,jnp.int32(t))
        if t-1 in [800,600,400,200,100,50,0]:
            stages[t-1]=np.asarray(x);print('generation level',t-1,'seconds',time.perf_counter()-begin,flush=True)
    levels=list(stages);fig,axs=plt.subplots(4,len(levels),figsize=(14,7))
    for j,t in enumerate(levels):
        for i,img in enumerate(display(stages[t])):
            axs[i,j].imshow(img,interpolation='nearest');axs[i,j].axis('off')
            if i==0:axs[i,j].set_title(f't={t}')
    savefig(fig,'photographs.png')
    spec=importlib.util.spec_from_file_location('week6_data',ROOT.parent/'week6/scripts/data.py');data=importlib.util.module_from_spec(spec);spec.loader.exec_module(data)
    real,labels=data.cifar('test');data.EVIDENCE=E
    data.gallery(np.concatenate([display(x),real[:4]/255.]),labels=['Generated 1','Generated 2','Generated 3','Generated 4']+[str(data.CIFAR_CLASSES[i]) for i in labels[:4]],columns=4,filename='cifar_comparison.png',title='Generated (top) and real CIFAR-10 test images (bottom)',vmin=0,vmax=1)
    np.savez(A/'photographs.npz',**{f't{t}':z for t,z in stages.items()})
    result=dict(seed=7200,steps=1000,batch=4,dtype='float32',model_index='t-1',clipping='display only',seconds=time.perf_counter()-begin,final_min=float(x.min()),final_max=float(x.max()),revision=REVISION)
    (E/'photographs.json').write_text(json.dumps(result,indent=2))

def repaint(fn,w,observations,mask,seed):
    """Only observed values and mask enter this function; no hidden truth."""
    key=jax.random.PRNGKey(seed);x=jax.random.normal(key,observations.shape)
    @jax.jit
    def attempt(w,x,t,tp,draw):
        a=AB[t]/AB[tp];eps=fn(w,x,t-1)
        clean=jnp.clip((x-jnp.sqrt(1-AB[t])*eps)/jnp.sqrt(AB[t]),-1,1)
        mean=jnp.sqrt(AB[tp])*(1-a)/(1-AB[t])*clean+jnp.sqrt(a)*(1-AB[tp])/(1-AB[t])*x
        var=(1-AB[tp])*(1-a)/(1-AB[t])
        xi=jax.random.normal(jax.random.fold_in(key,draw),x.shape);eta=jax.random.normal(jax.random.fold_in(key,draw+1),x.shape)
        proposal=mean+jnp.sqrt(var)*xi;known=jnp.sqrt(AB[tp])*observations+jnp.sqrt(1-AB[tp])*eta
        return jnp.where(mask,known,proposal)
    evaluations=0;draw=1
    for t in range(1000,0,-20):
        tp=t-20;a=AB[t]/AB[tp]
        for r in range(2 if tp else 1):
            if r:
                f=jax.random.normal(jax.random.fold_in(key,draw),x.shape);draw+=1;x=jnp.sqrt(a)*x+jnp.sqrt(1-a)*f
            x=attempt(w,x,jnp.int32(t),jnp.int32(tp),jnp.int32(draw));draw+=2;evaluations+=1
    return np.asarray(x),evaluations

def inpaint():
    config,w=load_model();fn=predictor(config);raw=np.load(ROOT/'inputs/images.npz');print('input keys',raw.files,flush=True)
    truth=raw['test'].astype(np.float32)
    if truth.shape==(4,32,32,3):truth=truth.transpose(0,3,1,2)
    assert truth.shape==(4,3,32,32) and truth.min()>=-1 and truth.max()<=1
    mask=np.ones_like(truth,dtype=bool);mask[:,:,6:26,6:26]=False;observed=np.where(mask,truth,0)
    completions=[];start=time.perf_counter();counts=[]
    for seed in [7201,7202,7203,7204]:
        x,count=repaint(fn,w,jnp.array(observed),jnp.array(mask),seed);completions.append(x);counts.append(count)
        print('RePaint seed',seed,'seconds',time.perf_counter()-start,flush=True)
    completions=np.array(completions);mse=((completions-truth)**2)[:,:,:,6:26,6:26].mean(axis=(2,3,4))
    variance=completions.var(axis=0);change=float(abs(completions[:,mask]-truth[mask]).max())
    fig,axs=plt.subplots(4,7,figsize=(14,8))
    for i in range(4):
        views=[display(truth)[i],display(observed)[i]]+[display(c)[i] for c in completions]
        titles=['Truth','Observed']+[f'Seed {7201+j}\nMSE {mse[j,i]:.3f}' for j in range(4)]
        for j,(view,title) in enumerate(zip(views,titles)):
            axs[i,j].imshow(view,interpolation='nearest');axs[i,j].set_title(title,fontsize=9);axs[i,j].axis('off')
        im=axs[i,6].imshow(variance[i].mean(0),vmin=0,vmax=variance.mean(1).max(),cmap='magma',interpolation='nearest');axs[i,6].set_title('Pixel variance');axs[i,6].axis('off');fig.colorbar(im,ax=axs[i,6],shrink=.7)
    savefig(fig,'inpainting.png')
    result=dict(seeds=[7201,7202,7203,7204],mse_seed_by_image=mse.tolist(),mean_mse=float(mse.mean()),max_observed_change=change,observed_variance_max=float(variance[mask].max()),hidden_variance_mean=float(variance[~mask].mean()),network_evaluations_per_seed=counts,seconds=time.perf_counter()-start)
    np.savez(A/'inpainting.npz',truth=truth,observed=observed,mask=mask,completions=completions,variance=variance)
    (E/'inpainting.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('task',choices=['verify','generate','inpaint']);args=parser.parse_args();globals()[args.task]()
