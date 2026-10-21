"""Week 7 Parts 1–3: learned transport, with independent analytic references."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from pathlib import Path
import json, time, argparse
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial.distance import cdist, pdist

ROOT = Path(__file__).resolve().parents[1]
A, E = ROOT/'artifacts', ROOT/'evidence'
A.mkdir(exist_ok=True); E.mkdir(exist_ok=True)
CENTRES = jnp.array([[3.,0.],[0.,3.],[-3.,0.],[0.,-3.]])
BETA = jnp.linspace(1e-4, .02, 1000)
AB = jnp.concatenate([jnp.ones(1), jnp.cumprod(1-BETA)])
FREQ = jnp.geomspace(1., 100., 16)
STAGES = [0,100,250,500,1000,2500,5000,10000,17500,25000]

def exact_data(key, n):
    k1,k2=jax.random.split(key)
    return CENTRES[jax.random.randint(k1,(n,),0,4)]+.3*jax.random.normal(k2,(n,2))

def init(key):
    dims=[34,128,128,128,2]
    return [(jax.random.normal(k,(a,b))*jnp.sqrt(2/a),jnp.zeros(b))
            for k,a,b in zip(jax.random.split(key,4),dims[:-1],dims[1:])]

def net(params,x,s):
    s=jnp.broadcast_to(jnp.asarray(s),x.shape[:-1])
    phase=s[...,None]*FREQ
    h=jnp.concatenate([x,jnp.sin(phase),jnp.cos(phase)],axis=-1)
    for w,b in params[:-1]: h=jax.nn.silu(h@w+b)
    w,b=params[-1]
    return h@w+b

def exact_noise(x,t):
    ab=AB[t]; var=.09*ab+1-ab
    delta=x[...,None,:]-jnp.sqrt(ab)*CENTRES
    weights=jax.nn.softmax(-jnp.sum(delta**2,axis=-1)/(2*var),axis=-1)
    return jnp.sqrt(1-ab)/var*jnp.sum(weights[...,None]*delta,axis=-2)

def exact_velocity(x,s):
    var=.09*(1-s)**2+s*s
    delta=x[...,None,:]-(1-s)*CENTRES
    weights=jax.nn.softmax(-jnp.sum(delta**2,axis=-1)/(2*var),axis=-1)
    component=-CENTRES+(s-.09*(1-s))/var*delta
    return jnp.sum(weights[...,None]*component,axis=-2)

def train(kind, updates=25000):
    seed=7101 if kind=='noise' else 7102
    params=init(jax.random.PRNGKey(seed)); key=jax.random.PRNGKey(seed+100)
    schedule=optax.linear_schedule(1e-3,0.,updates-1)
    optimizer=optax.adam(schedule); state=optimizer.init(params)
    @jax.jit
    def block(params,state,key,indices):
        def step(carry,j):
            p,st,k=carry; k,kd,kt,kz=jax.random.split(k,4)
            data=exact_data(kd,512); z=jax.random.normal(kz,data.shape)
            if kind=='noise':
                t=jax.random.randint(kt,(512,),1,1001); s=t/1000.
                x=jnp.sqrt(AB[t,None])*data+jnp.sqrt(1-AB[t,None])*z; target=z
            else:
                s=jax.random.uniform(kt,(512,)); x=(1-s[:,None])*data+s[:,None]*z; target=z-data
            loss,grad=jax.value_and_grad(lambda p:jnp.mean((net(p,x,s)-target)**2))(p)
            upd,st=optimizer.update(grad,st,p); p=optax.apply_updates(p,upd)
            return (p,st,k),loss
        return jax.lax.scan(step,(params,state,key),indices)
    stages=STAGES if updates==25000 else sorted(set(STAGES[:-1]+[updates]))
    losses=[]; start=time.perf_counter()
    for i,end in enumerate(stages):
        if i:
            (params,state,key),ls=block(params,state,key,jnp.arange(stages[i-1],end))
            losses.extend(np.asarray(ls).tolist())
        np.savez(A/f'{kind}_{end}.npz',**{f'{i}_{name}':np.asarray(v) for i,p in enumerate(params) for name,v in zip(['w','b'],p)})
        print(kind,end, np.mean(losses[-100:]) if losses else None,flush=True)
    settings=dict(seed=seed,updates=updates,batch=512,architecture=[34,128,128,128,2],activation='silu',time_features='sin/cos(geomspace(1,100,16)*s)',optimizer='Adam',learning_rate_initial=.001,learning_rate_final=0.,stages=stages,seconds=time.perf_counter()-start)
    np.savez(A/f'{kind}_training.npz',loss=losses,settings=json.dumps(settings))
    (E/f'{kind}_training.json').write_text(json.dumps(settings,indent=2))

def load(kind,stage=25000):
    d=np.load(A/f'{kind}_{stage}.npz')
    return [(jnp.array(d[f'{i}_w']),jnp.array(d[f'{i}_b'])) for i in range(4)]

@partial(jax.jit,static_argnames=('method','steps','analytic','trace'))
def sample(params,start,key,method='ddpm',steps=1000,analytic=False,trace=False):
    levels=jnp.rint(jnp.linspace(1000,0,steps+1)).astype(jnp.int32)
    def step(x,item):
        i,t,tp=item
        if method=='flow':
            s=1-i/steps
            velocity=exact_velocity(x,s) if analytic else net(params,x,s)
            y=x-velocity/steps
        else:
            eps=exact_noise(x,t) if analytic else net(params,x,t/1000.)
            if method=='ddim':
                clean=(x-jnp.sqrt(1-AB[t])*eps)/jnp.sqrt(AB[t])
                y=jnp.sqrt(AB[tp])*clean+jnp.sqrt(1-AB[tp])*eps
            else:
                a=AB[t]/AB[tp]
                mean=(x-(1-a)/jnp.sqrt(1-AB[t])*eps)/jnp.sqrt(a)
                noise=jax.random.normal(jax.random.fold_in(key,i),x.shape)
                y=mean+jnp.where(tp>0,jnp.sqrt(1-a),0)*noise
        return y,y if trace else None
    end,path=jax.lax.scan(step,start,(jnp.arange(steps),levels[:-1],levels[1:]))
    return (end,jnp.concatenate([start[None],path])) if trace else end

def metrics(x):
    x=np.asarray(x); centres=np.asarray(CENTRES)
    labels=np.argmin(np.sum((x[:,None]-centres)**2,axis=-1),axis=1)
    fractions=np.bincount(labels,minlength=4)/len(x)
    means=np.array([x[labels==i].mean(0) if np.any(labels==i) else [np.nan]*2 for i in range(4)])
    widths=np.array([x[labels==i].std(0) if np.any(labels==i) else [np.nan]*2 for i in range(4)])
    return dict(fractions=fractions.tolist(),centres=means.tolist(),widths=widths.tolist(),fraction_error=float(abs(fractions-.25).max()),centre_error=float(abs(means-centres).max()),centre_l2_error=float(np.linalg.norm(means-centres,axis=1).max()),width_error=float(abs(widths-.3).max()))

def energy(x,y):
    # Unbiased U-statistic: omit within-sample diagonal and count unordered pairs.
    x,y=np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64)
    return float(2*cdist(x,y).mean()-pdist(x).mean()-pdist(y).mean())

def scatter(ax,x,title):
    x=np.asarray(x); ax.scatter(x[:,0],x[:,1],s=2,alpha=.35)
    ax.scatter(*np.asarray(CENTRES).T,marker='+',c='red',s=40)
    ax.set(xlim=(-5,5),ylim=(-5,5),aspect='equal',title=title,xlabel='x1',ylabel='x2')

def savefig(fig,name):
    fig.tight_layout(); fig.savefig(E/name,dpi=150); plt.close(fig)

def movie(path,levels,name):
    fig,ax=plt.subplots(figsize=(5,5)); dots=ax.scatter([],[],s=3,alpha=.4)
    ax.set(xlim=(-5,5),ylim=(-5,5),aspect='equal',xlabel='x1',ylabel='x2')
    def update(i):
        dots.set_offsets(path[i]); ax.set_title(f'{name}: t={levels[i]}'); return dots,
    FuncAnimation(fig,update,frames=len(path),interval=100).save(E/f'{name}.gif',writer=PillowWriter(fps=10)); plt.close(fig)

def part1():
    p=load('noise'); key=jax.random.PRNGKey(7110)
    data=exact_data(key,2000)
    @jax.jit
    def forward(x):
        def step(x,t):
            y=jnp.sqrt(1-BETA[t])*x+jnp.sqrt(BETA[t])*jax.random.normal(jax.random.fold_in(key,t+1),x.shape)
            return y,y
        return jax.lax.scan(step,x,jnp.arange(1000))[1]
    fp=np.concatenate([np.asarray(data)[None],np.asarray(forward(data))])
    indices=np.arange(0,1001,20); movie(fp[indices],indices,'forward')
    start=jax.random.normal(jax.random.PRNGKey(7111),(2000,2))
    end,rp=sample(p,start,jax.random.PRNGKey(7112),trace=True); end,rp=np.asarray(end),np.asarray(rp)
    movie(rp[indices],1000-indices,'reverse')
    fig,axs=plt.subplots(1,2,figsize=(10,4)); scatter(axs[0],data,'Exact target'); scatter(axs[1],end,'Learned DDPM: 1000 steps'); savefig(fig,'ddpm_distribution.png')
    paths=[]
    for stage in STAGES:
        _,path=sample(load('noise',stage),start[:10],jax.random.PRNGKey(7113),trace=True)
        paths.append(np.asarray(path))
    paths=np.array(paths); loss=np.load(A/'noise_training.npz')['loss']
    fig,(ax,al)=plt.subplots(1,2,figsize=(10,4))
    def frame(i):
        ax.clear(); al.clear(); stage=STAGES[i]
        for k in range(10):
            ax.plot(*paths[i,:,k].T,lw=.6); ax.scatter(*paths[i,-1,k],s=18)
        for c in np.asarray(CENTRES):ax.add_patch(plt.Circle(c,.6,fill=False,ls='--',color='grey'))
        ax.set(xlim=(-5,5),ylim=(-5,5),aspect='equal',title=f'Update {stage}; fixed starts/noise')
        for k in range(10): ax.scatter(*paths[i,0,k],s=20,facecolors='white',edgecolors='black')
        n=stage//100
        if n:al.plot(np.arange(n)*100+50,loss[:n*100].reshape(n,100).mean(1))
        al.set(xlim=(0,25000),ylim=(0,1.5),xlabel='Training updates',ylabel='Noise MSE (100-update mean)',title='Training history so far')
        fig.tight_layout()
    FuncAnimation(fig,frame,frames=10,interval=800).save(E/'training.gif',writer=PillowWriter(fps=1.25)); frame(9); fig.savefig(E/'training.png',dpi=150); plt.close(fig)
    summary=dict(forward_mean=fp[-1].mean(0).tolist(),forward_variance=fp[-1].var(0).tolist(),ddpm=metrics(end),final_1000_loss=float(loss[-1000:].mean()),initial_path_max=float(abs(paths[0]).max()),ten_endpoint_distances=np.min(np.linalg.norm(paths[-1,-1,:,None]-np.asarray(CENTRES),axis=-1),axis=1).tolist())
    np.savez(A/'part1.npz',data=data,forward=fp[indices],forward_levels=indices,reverse=rp[indices],reverse_levels=1000-indices,end=end,stage_paths=paths,start=start)
    (E/'part1.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)

def part2():
    p=load('noise'); start=jnp.array(np.load(A/'part1.npz')['start']); key=jax.random.PRNGKey(7120)
    outputs={}; paths={}
    for name,method,k in [('DDIM A','ddim',0),('DDIM B','ddim',1),('DDPM A','ddpm',2),('DDPM B','ddpm',3)]:
        end,path=sample(p,start,jax.random.fold_in(key,k),method=method,trace=True)
        outputs[name]=np.asarray(end); paths[name]=np.asarray(path)[:,:10]
    exact=np.asarray(exact_data(jax.random.PRNGKey(7121),2000))
    fig,axs=plt.subplots(2,3,figsize=(14,9))
    scatter(axs[0,0],exact,'Exact target')
    for ax,name in zip([axs[0,1],axs[0,2],axs[1,1],axs[1,2]],outputs):scatter(ax,outputs[name],name)
    ax=axs[1,0]
    for name,style in [('DDIM A','-'),('DDPM A','--'),('DDPM B',':')]:
        ax.plot(*paths[name][:,0].T,style,label=name,lw=1)
        ax.scatter(*outputs[name][0],s=35)
    ax.set(xlim=(-5,5),ylim=(-5,5),aspect='equal',title='One shared Gaussian start'); ax.legend()
    savefig(fig,'determinism.png')
    labels=lambda x:np.argmin(np.sum((x[:,None]-np.asarray(CENTRES))**2,axis=-1),axis=1)
    result={name:metrics(x) for name,x in outputs.items()}
    result['ddim_max_difference']=float(abs(outputs['DDIM A']-outputs['DDIM B']).max())
    result['ddpm_same_cluster_fraction']=float(np.mean(labels(outputs['DDPM A'])==labels(outputs['DDPM B'])))
    np.savez(A/'part2.npz',**outputs,**{name+'_paths':v for name,v in paths.items()})
    (E/'part2.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

def part3():
    p=load('noise'); v=load('flow'); results={}; arrays={}; start=jax.random.normal(jax.random.PRNGKey(7130),(4000,2))
    target=np.asarray(exact_data(jax.random.PRNGKey(7131),4000)); null=[]
    for i in range(20):null.append(energy(exact_data(jax.random.PRNGKey(7140+2*i),4000),exact_data(jax.random.PRNGKey(7141+2*i),4000)))
    for analytic in [False,True]:
        for method in ['ddpm','ddim','flow']:
            for steps in [1,2,5,10,20,50,100]+([] if method=='flow' else [1000]):
                name=f'{method}_{"exact" if analytic else "learned"}_{steps}'
                x=np.asarray(sample(v if method=='flow' else p,start,jax.random.PRNGKey(7190),method=method,steps=steps,analytic=analytic))
                results[name]=metrics(x);results[name]['energy_distance']=energy(x,target);arrays[name]=x
                print(name,results[name]['energy_distance'],results[name]['width_error'],flush=True)
    field={}; fig,axs=plt.subplots(1,3,figsize=(15,4))
    for ax,s in zip(axs,[.1,.5,.9]):
        test=(1-s)*exact_data(jax.random.PRNGKey(7181),10000)+s*jax.random.normal(jax.random.PRNGKey(7182),(10000,2))
        truth=np.asarray(exact_velocity(test,s)); pred=np.asarray(net(v,test,s))
        error=float(np.sqrt(np.sum((truth-pred)**2)/np.sum(truth**2)));field[str(s)]=error
        xx,yy=np.meshgrid(np.linspace(-4,4,17),np.linspace(-4,4,17)); grid=np.c_[xx.ravel(),yy.ravel()]
        vt=np.asarray(exact_velocity(grid,s));vp=np.asarray(net(v,grid,s))
        # Arrows on the grid are visual context; RMS is on independent marginal samples.
        for vv,color,label in [(vt,'black','Exact'),(vp,'tab:orange','Learned')]:
            vv=vv/np.maximum(1,np.linalg.norm(vv,axis=1,keepdims=True)/3)
            ax.quiver(*grid.T,*vv.T,color=color,alpha=.65,scale=50,label=label)
        ax.set(aspect='equal',title=f's={s}, sample RMS error={error:.4f}');ax.legend()
        arrays[f'field_{s}_points']=np.asarray(test);arrays[f'field_{s}_exact']=truth;arrays[f'field_{s}_learned']=pred
    savefig(fig,'velocity.png')
    fig,axs=plt.subplots(1,3,figsize=(15,4));q=float(np.quantile(null,.95))
    for method,color in [('ddpm','tab:blue'),('ddim','tab:orange'),('flow','tab:green')]:
        steps=[1,2,5,10,20,50,100]+([] if method=='flow' else [1000])
        for analytic,ls in [(False,'-'),(True,'--')]:
            r=[results[f'{method}_{"exact" if analytic else "learned"}_{n}'] for n in steps]
            for ax,metric in zip(axs,['energy_distance','fraction_error','width_error']):
                y=[d[metric] for d in r]
                if metric=='energy_distance':y=np.maximum(y,1e-4)
                ax.plot(steps,y,ls,marker='.',color=color,label=method+(' exact' if analytic else ' learned'))
    axs[0].axhspan(1e-4,max(q,1e-4),color='grey',alpha=.2);axs[0].set_yscale('log')
    axs[1].axhspan(0,2*np.sqrt(.25*.75/4000),color='grey',alpha=.2)
    axs[2].axhline(.03,color='grey',ls=':')
    for ax,title in zip(axs,['Energy distance (floor 1e-4)','Largest cluster fraction error','Largest coordinate width error']):
        ax.set(xscale='log',xlabel='Network evaluations',title=title);ax.axvline(20,color='grey',ls=':')
    axs[0].legend(fontsize=7);savefig(fig,'step_comparison.png')
    results.update(null_distances=null,null_95=q,velocity_relative_rms=field,sample_count=4000)
    np.savez(A/'part3.npz',start=start,target=target,**arrays)
    (E/'part3.json').write_text(json.dumps(results,indent=2)); print('velocity',field,'null',q,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('task',choices=['train-noise','train-flow','part1','part2','part3']);args=parser.parse_args()
    if args.task.startswith('train-'):train(args.task.split('-')[1])
    else:globals()[args.task]()
