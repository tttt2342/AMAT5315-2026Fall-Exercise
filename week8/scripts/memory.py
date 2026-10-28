from common import *
import time
from PIL import Image,ImageDraw,ImageFont

def unit(r,shape):
 x=r.normal(size=shape); return x/np.linalg.norm(x,axis=-1,keepdims=True)
def ratios(k,v):
 """Query ALL stored keys; chunk nonlinear scores to avoid M x M allocation."""
 m,d=k.shape; outer=k@(k.T@v); r0=np.linalg.norm(outer-v,axis=1)
 pinv=np.linalg.pinv(k); recalled=k@(pinv@v); own=np.einsum('ij,ji->i',k,pinv)
 rls=np.linalg.norm(recalled-own[:,None]*v,axis=1)/own
 sums=np.zeros(3)
 for start in range(0,m,512):
  stop=min(start+512,m); sim=np.maximum(k[start:stop]@k.T,0); weights=sim.copy()
  for power in range(3):
   out=weights@v; sums[power]+=np.linalg.norm(out-v[start:stop],axis=1).sum(); weights*=sim
 return np.r_[r0.mean(),rls.mean(),sums/m]
def capacity(ms,rs):
 if rs[0]>=1: return float(ms[0]) # Lower-censored; report separately.
 j=np.flatnonzero(rs>=1)
 if not len(j): return None
 i=j[0]; return float(np.exp(np.interp(0,np.log(rs[i-1:i+1]),np.log(ms[i-1:i+1]))))
def main():
 r=np.random.default_rng(1830); ms=2**np.arange(3,17); dims=[16,32,64]; raw=np.zeros((3,3,len(ms),5)); start=time.monotonic()
 for di,d in enumerate(dims):
  for rep in range(3):
   for mi,m in enumerate(ms):
    k=unit(r,(m,d)); v=unit(r,(m,d)); raw[di,rep,mi]=ratios(k,v)
    print(d,rep,m,raw[di,rep,mi].round(4),round(time.monotonic()-start,1),flush=True)
    np.savez(ART/'memory_progress.npz',raw=raw,dims=dims,ms=ms)
 means=np.median(raw,axis=1); caps=np.array([[capacity(ms,means[i,:,j]) for j in range(5)] for i in range(3)])
 names=['outer','least squares','ReLU','ReLU²','ReLU³']; slopes=[float(np.polyfit(np.log(dims),np.log(caps[:,j]),1)[0]) for j in range(5)]
 save('memory',{'seed':1830,'draws':3,'aggregation':'arithmetic mean of per-query R, then median over 3 independent draws','dimensions':dims,'M':ms,'stores':names,'ratios':means,'capacities':caps,'capacity_slopes':slopes,'seconds':time.monotonic()-start,'raw':'artifacts/memory_progress.npz','least_squares_max_R_M_le_d':float(max(raw[i,:,ms<=d,1].max() for i,d in enumerate(dims)))})
 plt.figure(figsize=(10,4)); ax=plt.subplot(121)
 for j,name in enumerate(names): ax.loglog(ms,means[2,:,j],'o-',label=name)
 for theory in [np.sqrt(ms/64),np.sqrt(np.maximum(ms/64-1,1e-12)),np.sqrt(ms/128),np.sqrt(3*ms/(2*64**2)),np.sqrt(15*ms/(2*64**3))]: ax.loglog(ms,theory,'--',alpha=.35,color='gray')
 ax.axhline(1,color='black',ls=':'); ax.set(xlabel='stored memories M',ylabel='interference / signal',ylim=(.002,100)); ax.legend(fontsize=8)
 ax=plt.subplot(122)
 for j,name in enumerate(names): ax.loglog(dims,caps[:,j],'o-',label=f'{name}: {slopes[j]:.2f}')
 ax.set(xlabel='dimension d',ylabel='capacity (R = 1)'); ax.legend(fontsize=8); fig('memory')
def pictures():
 img=Image.new('L',(32,32),255); draw=ImageDraw.Draw(img); font=ImageFont.truetype('/System/Library/Fonts/STHeiti Medium.ttc',29); box=draw.textbbox((0,0),'\u7b97',font=font); draw.text(((32-(box[2]-box[0]))/2-box[0],(32-(box[3]-box[1]))/2-box[1]),'\u7b97',font=font,fill=0)
 target=np.where(np.array(img)<128,1.,-1.).ravel(); r=np.random.default_rng(1840); ms=[64,256,1024,4096]; names=['outer','ReLU','ReLU²','softmax β=8']; ratios_out=np.zeros((20,4,4)); shown=[]
 for rep in range(20):
  for mi,m in enumerate(ms):
   k=unit(r,(m,64)); values=r.choice([-1.,1.],(m,1024)); values[0]=target; s=k@k[0]; weights=[s,np.maximum(s,0),np.maximum(s,0)**2,np.exp(8*(s-s.max()))]
   recalls=[]
   for j,w in enumerate(weights):
    y=(w@values)/w[0]; ratios_out[rep,mi,j]=np.linalg.norm(y-target)/np.linalg.norm(target); recalls.append(y)
   if rep==0: shown.append(recalls)
 yy,xx=np.mgrid[:32,:32]; plt.figure(figsize=(10,10))
 for mi,m in enumerate(ms):
  for j,name in enumerate(names):
   ax=plt.subplot(4,4,mi*4+j+1); y=shown[mi][j]; ax.scatter(xx.ravel(),yy.ravel(),s=np.maximum(y,0)*2,c='#b44b42' if ratios_out[0,mi,j]<1 else '#555555',linewidths=0); ax.set_aspect('equal'); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(f'R={ratios_out[0,mi,j]:.3f}');
   if mi==0: ax.set_title(name)
   if j==0: ax.set_ylabel(f'M={m}')
 fig('pictures'); np.savez(ART/'pictures.npz',target=target,recalls=shown,ratios=ratios_out)
 save('pictures',{'character_codepoint':'U+7B97','font':'STHeiti Medium','seed':1840,'M':ms,'stores':names,'ratios_first_draw':ratios_out[0],'ratios_median_20':np.median(ratios_out,axis=0),'sharpness':8,'dimensions':64,'image_shape':[32,32]})
if __name__=='__main__':
 import sys
 if sys.argv[1]=='pictures': pictures()
 else: main()
