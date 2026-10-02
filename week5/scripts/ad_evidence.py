import sys, json, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.ad import energy, passes
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
OUT=Path('artifacts/ad'); OUT.mkdir(parents=True,exist_ok=True)
def save(name):
    plt.savefig(OUT/name,dpi=150,bbox_inches='tight'); plt.close()
def graph(fun,name):
    jp=jax.make_jaxpr(fun)(1.3).jaxpr
    g=nx.DiGraph(); labels={}; producers={}; layers={}
    for i,v in enumerate(jp.invars):
        k=f'in{i}'; g.add_node(k); labels[k]='r'; producers[str(v)]=k; layers[k]=0
    for i,eq in enumerate(jp.eqns):
        k=f'op{i}'; g.add_node(k); labels[k]=eq.primitive.name+(' '+str(eq.params) if eq.params else '')
        sources=[]
        for j,v in enumerate(eq.invars):
            if str(v) in producers: q=producers[str(v)]
            else:
                q=f'lit{i}_{j}'; g.add_node(q); labels[q]=str(getattr(v,'val',v)); layers[q]=0
            g.add_edge(q,k); sources.append(q)
        layers[k]=max(layers[q] for q in sources)+1
        for v in eq.outvars: producers[str(v)]=k
    for v in jp.outvars:
        g.add_edge(producers[str(v)],'output'); labels['output']='output'; layers['output']=max(layers.values())+1
    pos={}
    for layer in set(layers.values()):
        nodes=[k for k in g if layers[k]==layer]
        for j,k in enumerate(nodes): pos[k]=(j-(len(nodes)-1)/2,-layer)
    plt.figure(figsize=(12,10 if len(jp.eqns)>5 else 6))
    nx.draw_networkx(g,pos,labels=labels,node_size=2100,node_color='#dcebf4',font_size=8,arrowsize=15)
    plt.axis('off'); save(name)
    return jp
u,t,a=passes(1.3)
rpt={'r':1.3,'energy':float(u),'tangents':{k:float(v) for k,v in t.items()},'adjoints':{k:float(v) for k,v in a.items()},'jax_grad':float(jax.grad(energy)(1.3))}
(OUT/'derivatives.json').write_text(json.dumps(rpt,indent=2)+'\n')
assert abs(float(u)+.6570169144600471)<1e-12
assert all(abs(float(v)-2.239979929791143)<1e-12 for v in [t['U'],a['r'],rpt['jax_grad']])
assert abs(a['a']+2.3425903117359734)<1e-12
r=jnp.linspace(.95,2.5,601); exact=24*(r**-7-2*r**-13)
_,t,a=passes(r); fd=(energy(r+1e-6)-energy(r-1e-6))/(2e-6)
fig,axs=plt.subplots(1,2,figsize=(11,4))
errs={}
for label,y in [('analytic',exact),('forward',t['U']),('reverse',a['r']),('finite difference',fd)]:
    axs[0].plot(r,y,label=label)
    if label!='analytic':
        err=np.abs(y-exact); errs[label]=float(err.max()); axs[1].semilogy(r,np.maximum(err,1e-16),label=label)
for ax in axs: ax.set_xlabel('Separation r'); ax.legend()
axs[0].set_ylabel('dU/dr'); axs[0].axhline(0,color='gray',lw=.5); axs[1].set_ylabel('Absolute error'); save('modes.png')
assert errs['forward']<1e-12 and errs['reverse']<1e-12
assert max(errs['forward'],errs['reverse'])<errs['finite difference']
graph(energy,'graph.png'); jp=graph(jax.grad(energy),'grad-graph.png'); assert any(e.primitive.name=='add_any' for e in jp.eqns)
print('Maximum derivative errors:',errs,flush=True)
def timing(f,*args,repeat=5):
    f(*args).block_until_ready()
    ts=[]
    for _ in range(repeat):
        start=time.perf_counter(); f(*args).block_until_ready(); ts.append(time.perf_counter()-start)
    return float(np.median(ts))
rows=[]
for n in [64,128,256,512,1024]:
    rng=np.random.default_rng(5315+n); side=int(np.ceil(n**(1/3)))
    xyz=np.indices((side,side,side)).reshape(3,-1).T[:n]*2**(1/6)+rng.normal(0,.05,(n,3))
    ii,jj=np.triu_indices(n,1); i=jnp.array(ii); j=jnp.array(jj)
    def total(x):
        d=x.reshape(n,3)[i]-x.reshape(n,3)[j]; a=jnp.sum(d*d,axis=1)**-3
        return jnp.sum(4*(a*a-a))
    e=jax.jit(total); rev=jax.jit(jax.grad(total))
    # Exactly one input direction per JVP, executed sequentially by lax.map.
    def forward(x):
        return jax.lax.map(lambda k:jax.jvp(total,(x,),(jax.nn.one_hot(k,x.size,dtype=x.dtype),))[1],jnp.arange(x.size))
    fw=jax.jit(forward); x=jnp.array(xyz.reshape(-1))
    d=xyz[ii]-xyz[jj]; r2=(d*d).sum(axis=1); pair=24*(r2**-4-2*r2**-7)[:,None]*d
    analytic=np.zeros_like(xyz); np.add.at(analytic,ii,pair); np.add.at(analytic,jj,-pair)
    er={key:float(np.max(np.abs(np.array(f(x)).reshape(n,3)-analytic))/np.max(np.abs(analytic))) for key,f in [('forward',fw),('reverse',rev)]}
    te=timing(e,x,repeat=9); tr=timing(rev,x,repeat=9); tf=timing(fw,x,repeat=3)
    row=dict(N=n,P=3*n,energy_seconds=te,forward_ratio=tf/te,reverse_ratio=tr/te,relative_errors=er)
    rows.append(row); print(row,flush=True); assert max(er.values())<1e-12
p=np.array([r['P'] for r in rows]); plt.figure(figsize=(7,4))
for key in ['forward','reverse']: plt.loglog(p,[r[key+'_ratio'] for r in rows],'o-',label=key)
plt.loglog(p,p*rows[0]['forward_ratio']/p[0],':',label='proportional to P'); plt.xlabel('Inputs P = 3N'); plt.ylabel('Gradient time / energy time'); plt.legend(); save('scaling.png')
(OUT/'checks.json').write_text(json.dumps(dict(max_errors=errs,scaling=rows),indent=2)+'\n')
assert rows[-1]['forward_ratio']>100*rows[-1]['reverse_ratio']
