"""Independently recount serialized plans and draw their binary trees."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def recount(plan,cost):
    dims=plan['dimensions'];scopes=plan['input_scopes'];seen=[];steps=[]
    def visit(t):
        if 'Leaf' in t:
            i=t['Leaf']['tensor_index'];seen.append(i);return i,scopes[i],{i}
        node=t['Node'];a,b=[visit(x) for x in node['args']];labels=node['eins']['iy']
        assert node['eins']['ixs']==[a[1],b[1]]
        union=set(a[1])|set(b[1]);leaves=a[2]|b[2]
        outside=set(plan['output'])
        for i,s in enumerate(scopes):
            if i not in leaves:outside.update(s)
        assert union&outside<=set(labels)<=union
        work=int(np.prod([dims[i] for i in union],dtype=object));entries=int(np.prod([dims[i] for i in labels],dtype=object))
        step=dict(id=len(scopes)+len(steps),children=[a[0],b[0]],inputs=[a[1],b[1]],union=sorted(union),retained=labels,work=work,entries=entries)
        steps.append(step);return step['id'],labels,leaves
    _,out,_=visit(plan['tree']);assert out==plan['output'];assert sorted(seen)==list(range(len(scopes)))
    assert steps==cost['steps']
    assert sum(s['work'] for s in steps)==cost['work']
    largest=max([int(np.prod([dims[v] for v in s],dtype=object)) for s in scopes]+[s['entries'] for s in steps])
    assert largest==cost['largest_tensor']
    return dict(work=cost['work'],largest_tensor=largest,steps=len(steps))
def main():
    fig,axes=plt.subplots(2,2,figsize=(13,9),gridspec_kw={'height_ratios':[3,1]})
    checks={}
    for col,method in enumerate(['random','greedy']):
        plan=read(f'runs/{method}/order.json');cost=read(f'runs/{method}/complexity.json');checks[method]=recount(plan,cost)
        ax=axes[0,col];n=len(plan['input_scopes']);leaf_counter=[0]
        def draw(t):
            if 'Leaf'in t:
                i=t['Leaf']['tensor_index'];x=leaf_counter[0];leaf_counter[0]+=1
                ax.text(x,0,f'f{i}\n'+','.join(map(str,plan['input_scopes'][i])),ha='center',va='top',fontsize=8);return x,0,i
            a,b=[draw(c) for c in t['Node']['args']];x=(a[0]+b[0])/2;y=max(a[1],b[1])+1
            for child in (a,b):ax.plot([child[0],x],[child[1],y],color='#6b7c8c',lw=1)
            labels=t['Node']['eins']['iy'];entries=np.prod([plan['dimensions'][v] for v in labels],dtype=object)
            ax.text(x,y,','.join(map(str,labels)) or 'scalar',ha='center',va='center',fontsize=8,bbox=dict(facecolor='#ffddae' if entries==cost['largest_generated_tensor'] else '#e4eef6',edgecolor='none',pad=3))
            return x,y,None
        _,height,_=draw(plan['tree']);ax.set(xlim=(-.6,n-.4),ylim=(-1,height+.6),title=f'{method}: work {cost["work"]}, largest {cost["largest_tensor"]}');ax.axis('off')
        ax=axes[1,col];ax.bar(['Work (terms)','Largest tensor (entries)'],[cost['work'],cost['largest_tensor']],color=['#466d94','#d58c48'])
        ax.set_ylim(0,max(cost['work'],cost['largest_tensor'])*1.2)
    fig.suptitle('AD scalar evidence plans: orange marks largest generated tensors',fontsize=15);fig.tight_layout();fig.savefig(ROOT/'evidence/orders.png',dpi=170);plt.close(fig)
    assert read('runs/analysis/complexity.json')==read('runs/random/complexity.json')
    assert abs(read('runs/replay/run.json')['evidence_mass']/read('runs/greedy/run.json')['evidence_mass']-1)<1e-10
    (ROOT/'evidence/order_checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks)
if __name__=='__main__':main()
