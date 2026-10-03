import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path('artifacts');ref=np.load(out/'adjoint/image.npy');rows=[]
def audit(actions,N,budget):
    saved={0};work=None;grads=[];counts=dict(grad_order_errors=0,invalid_restores=0,budget_overruns=0,invalid_actions=0);calls=0;peak=1
    for a in actions:
        kind,n,count=a['action'],a['step'],a['saved_states']
        if kind=='restore':
            counts['invalid_restores']+=int(n not in saved);work=n
        elif kind=='call':
            counts['invalid_actions']+=int(work!=n);work=n+1;calls+=1
        elif kind=='store':
            counts['invalid_actions']+=int(n!=work or n in saved);saved.add(n)
        elif kind=='grad':
            counts['invalid_actions']+=int(n not in saved);grads.append(n)
        elif kind=='fetch':
            counts['invalid_actions']+=int(n not in saved or n==0);saved.discard(n)
        else: counts['invalid_actions']+=1
        counts['invalid_actions']+=int(count!=len(saved));counts['budget_overruns']+=int(len(saved)>budget+1);peak=max(peak,len(saved))
    expected=list(range(N-1,-1,-1));counts['grad_order_errors']=sum(a!=b for a,b in zip(grads,expected))+abs(len(grads)-N);counts['invalid_actions']+=int(saved!={0});assert not any(counts.values()),counts
    return dict(**counts,forward_calls=calls,peak=peak)
for b,expected_calls in zip([1,3,5,10],[28680,1695,990,642]):
    folder=out/f'checkpoint-{b}';s=json.loads((folder/'result.json').read_text())['statistics'];image=np.load(folder/'image.npy');err=float(np.linalg.norm(image-ref)/np.linalg.norm(ref));assert err<1e-9;audits=[]
    for shot,st in enumerate(s['per_shot']):
        actions=json.loads((folder/st['actions_file']).read_text());a=audit(actions,240,b);assert a['forward_calls']==st['scheduler_forward_calls']==expected_calls;assert a['peak']==st['peak_saved_states']==b+1;audits.append(a)
        if b==5 and shot==0:
            plt.figure(figsize=(10,4))
            for kind in ['store','restore','call','grad','fetch']:
                pts=[(i,a['step']) for i,a in enumerate(actions) if a['action']==kind];x,y=zip(*pts);plt.scatter(x,y,s=5,label=kind)
            plt.xlabel('Operation index');plt.ylabel('Timestep');plt.title('Treeverse: shot 0, five additional slots');plt.legend(ncol=5);plt.savefig(out/'checkpoint-actions.png',dpi=150,bbox_inches='tight');plt.close()
    rows.append(dict(budget=b,relative_l2_error=err,peak_saved_states=s['peak_saved_states'],peak_saved_bytes=s['peak_saved_bytes'],forward_calls_per_shot=expected_calls,audits=audits))
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained');bs=[r['budget'] for r in rows]
axs[0].semilogy(bs,[r['forward_calls_per_shot'] for r in rows],'o-',label='Treeverse');axs[0].axhline(240,ls='--',color='gray',label='Full history: 240');axs[0].set(ylabel='Forward steps per shot',xlabel='Additional checkpoint slots');axs[0].legend()
axs[1].plot(bs,[r['peak_saved_bytes'] for r in rows],'o-');axs[1].set(xlabel='Additional checkpoint slots',ylabel='Peak saved-state bytes');axs[1].set_title('Full history: 6,481,936 bytes (241 states)')
for r in rows: axs[1].annotate(f"{r['peak_saved_states']} states",(r['budget'],r['peak_saved_bytes']),xytext=(0,6),textcoords='offset points',ha='center')
fig.savefig(out/'checkpoint-work.png',dpi=150);plt.close(fig)
(out/'checkpoint-checks.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
if '--reflector-only' not in sys.argv:
    e=json.loads(Path('inputs/marmousi.json').read_text());s=json.loads((out/'marmousi-image/result.json').read_text())['statistics'];image=np.load(out/'marmousi-image/image.npy');data=np.load(out/'marmousi-born/born_data.npy');vnorm=float(np.linalg.norm(image));assert abs(vnorm/6.7037741e-4-1)<1e-4;assert s['peak_saved_states']==6 and s['peak_saved_bytes']==20788320
    audits=[audit(json.loads((out/'marmousi-image'/st['actions_file']).read_text()),e['steps'],5) for st in s['per_shot']]
    r=dict(image_l2=vnorm,relative_error=abs(vnorm/6.7037741e-4-1),peak_saved_states=6,peak_saved_bytes=s['peak_saved_bytes'],audits=audits);(out/'marmousi-checks.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
    unit=e['dx']*e['length_unit_m']/1000;extent=[0,(e['nx']-1)*unit,(e['nz']-1)*unit,0];fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ax,a,title,label in [(axs[0,0],np.array(e['background']),'Smoothed Marmousi background','Speed (km/s)'),(axs[0,1],np.array(e['perturbation']),'Short-wavelength perturbation','Velocity perturbation (km/s)'),(axs[1,1],image,'Raw RTM image, no depth gain','RTM amplitude (arbitrary units)')]:
        # A single symmetric display scale across all depths; clipping uses the
        # reference's stated scale and changes no stored image values.
        v=2e-5 if a is image else np.max(abs(a));kwargs={} if ax is axs[0,0] else dict(vmin=-v,vmax=v)
        p=ax.imshow(a,extent=extent,aspect='auto',cmap='viridis' if ax is axs[0,0] else 'RdBu_r',**kwargs);ax.set(xlabel='Horizontal position (km)',ylabel='Depth (km)',title=title);fig.colorbar(p,ax=ax,label=label)
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((8,0),6,3,fill=False,edgecolor='black',linestyle='--',linewidth=.8))
    shots=np.array(e['shots']);shot=int(np.argmin(abs(shots[:,0]*unit-10)));assert abs(shots[shot,0]*unit-10)<1e-9;rx=np.array(e['receivers'])[:,0]*unit
    p=axs[1,0].imshow(data[shot],extent=[rx.min(),rx.max(),e['steps']*e['dt']*e['time_unit_s'],e['dt']*e['time_unit_s']],aspect='auto',cmap='RdBu_r',vmin=-.001,vmax=.001);axs[1,0].set(xlabel='Receiver position (km)',ylabel='Time (s)',title='Born gather; source x = 10 km');fig.colorbar(p,ax=axs[1,0],label='Scattered pressure');fig.savefig(out/'marmousi.png',dpi=150);plt.close(fig)
