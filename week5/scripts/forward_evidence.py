import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('artifacts'); e=json.loads(Path('inputs/reflector.json').read_text()); unit=e['length_unit_m']/1000; dx=e['dx']*unit
c=np.array(e['background']);m=np.array(e['perturbation']);shots=np.array(e['shots'])*dx;recv=np.array(e['receivers'])*dx
extent=[0,(e['nx']-1)*dx,(e['nz']-1)*dx,0]
t=np.arange(e['steps'])*e['dt'];theta=np.pi*e['source_frequency']*(t-e['source_peak_time']);pulse=(1-2*theta**2)*np.exp(-theta**2)
fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained');ax=axs[0]
im=ax.imshow(c,extent=extent,vmin=1.5,vmax=2.1,cmap='viridis'); ax.contour(np.arange(e['nx'])*dx,np.arange(e['nz'])*dx,m,levels=[.075],colors='white');ax.scatter(*shots.T,marker='*',color='red',label='Shots');ax.scatter(*recv.T,marker='v',s=14,color='orange',label='Receivers');ax.legend();ax.set(xlabel='Horizontal position (km)',ylabel='Depth (km)',title='Geometry; reflector at 2.1 km');fig.colorbar(im,ax=ax,label='Background speed (km/s)')
axs[1].plot(t*e['time_unit_s'],pulse);axs[1].set(xlabel='Time (s)',ylabel='Source pulse g(t)',title='Ricker pulse: 0.8 Hz, peak 1.5 s');fig.savefig(OUT/'inputs.png',dpi=150);plt.close(fig)
data=np.load(OUT/'forward/traces.npy');norm=float(np.linalg.norm(data));assert abs(norm/11.574770-1)<1e-4
maxima=[];fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained');v=abs(data).max()
for shot,(d,ax) in enumerate(zip(data,axs)):
 n,k=np.unravel_index(abs(d).argmax(),d.shape); peak=float(abs(d[n,k]));maxima.append(dict(shot=shot,trace_index=int(n),receiver_index=int(k),receiver=e['receivers'][k],peak=peak));assert n==83;assert abs(peak/[.60809514,.59271397,.60809514][shot]-1)<1e-4;assert abs(e['receivers'][k][0]-e['shots'][shot][0])==1
 im=ax.imshow(d,aspect='auto',extent=[recv[:,0].min(),recv[:,0].max(),e['steps']*e['dt']*e['time_unit_s'],e['dt']*e['time_unit_s']],cmap='RdBu_r',vmin=-v,vmax=v);ax.set(xlabel='Receiver position (km)',ylabel='Time (s)',title=f'Shot {shot}, source x = {shots[shot,0]:.1f} km')
fig.colorbar(im,ax=axs,label='Pressure');fig.savefig(OUT/'forward/gathers.png',dpi=150);plt.close(fig)
r=json.loads((OUT/'forward/run.json').read_text());frame=r['recording']['steps'].index(150);direct=np.load(OUT/'forward/wavefield.npy')[frame];echo=np.load(OUT/'forward/echo.npy')[frame]
report=dict(traces_l2=norm,maxima=maxima,step150_direct_max=float(abs(direct).max()),step150_echo_max=float(abs(echo).max()),echo_ratio=float(abs(echo).max()/abs(direct).max()))
(OUT/'forward/checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
