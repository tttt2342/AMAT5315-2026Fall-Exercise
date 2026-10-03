import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
e=json.loads(Path('inputs/reflector.json').read_text());m=np.array(e['perturbation']);im=np.load('artifacts/adjoint/image.npy');d=np.load('artifacts/born/born_data.npy');left=float(np.sum(d*d));right=float(np.sum(m*im));err=abs(left-right)/max(abs(left),abs(right));assert err<1e-9
window=im[10:34,7:34];depth=np.arange(10,34)*.1;profile=np.linalg.norm(window,axis=1);peak=int(np.argmax(profile))+10;assert abs(peak-21)<=1
fig,axs=plt.subplots(1,3,figsize=(11,4),layout='constrained')
for ax,a,title,label in zip(axs[:2],[m[10:34,7:34],window],['Known perturbation','Raw RTM image'],['Velocity perturbation (km/s)','RTM amplitude (arbitrary units)']):
 v=abs(a).max();p=ax.imshow(a,extent=[.65,3.35,3.35,.95],cmap='RdBu_r',vmin=-v,vmax=v,aspect='auto');ax.set(xlabel='Horizontal position (km)',ylabel='Depth (km)',title=title);fig.colorbar(p,ax=ax,label=label);ax.axhline(2.1,color='k',ls='--',lw=.7)
axs[2].plot(profile,depth);axs[2].invert_yaxis();axs[2].axhline(2.1,color='k',ls='--',label='True: 2.1 km');axs[2].axhline(peak*.1,color='red',ls=':',label=f'Peak: {peak*.1:.1f} km');axs[2].set(xlabel='Row L2 norm',ylabel='Depth (km)',title='Image depth profile');axs[2].legend();fig.savefig('artifacts/adjoint/image.png',dpi=150)
r=dict(born_squared_l2=left,perturbation_image_inner_product=right,transpose_relative_error=err,peak_depth_km=peak*.1,depth_error_km=abs(peak-21)*.1)
Path('artifacts/adjoint/checks.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
