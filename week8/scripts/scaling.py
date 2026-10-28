from common import *
from scipy.optimize import minimize
from scipy.special import logsumexp
from itertools import product
import csv,time

def objective(theta,lnN,lnD,lnL):
 terms=np.stack([np.full_like(lnN,theta[0]),theta[1]-theta[3]*lnN,theta[2]-theta[4]*lnD],axis=1); pred=logsumexp(terms,axis=1); r=pred-lnL; delta=.001
 # Divide objective and gradient by delta for reliable numerical convergence; same minimizer.
 loss=np.where(abs(r)<=delta,.5*r*r,delta*(abs(r)-delta/2)).mean()/delta
 influence=np.clip(r,-delta,delta)/delta/len(r); w=np.exp(terms-pred[:,None]); g=np.array([np.sum(influence*w[:,0]),np.sum(influence*w[:,1]),np.sum(influence*w[:,2]),-np.sum(influence*w[:,1]*lnN),-np.sum(influence*w[:,2]*lnD)])
 return loss,g

def fit(theta,N,D,L):
 return minimize(objective,theta,args=(np.log(N),np.log(D),np.log(L)),jac=True,method='L-BFGS-B',bounds=[(-5,2),(-5,20),(-5,25),(.01,1),(.01,1)],options={'ftol':1e-13,'gtol':1e-9,'maxiter':2000,'maxls':50})
def unpack(t): return np.r_[np.exp(t[:3]),t[3:]]
def law(p,N,D):
 E,A,B,a,b=p; return E+A/N**a+B/D**b

def optimum(p,C):
 E,A,B,a,b=p; G=(a*A/(b*B))**(1/(a+b)); N=G*(C/6)**(b/(a+b)); return N,C/(6*N)
def main():
 start=time.monotonic(); rows=list(csv.DictReader((DATA/'chinchilla-runs.csv').open())); N=np.array([float(r['Model Size']) for r in rows]); C=np.array([float(r['Training FLOP']) for r in rows]); L=np.array([float(r['loss']) for r in rows]); keep=np.argsort(L)[:-5]; removed=np.argsort(L)[-5:]; N,C,L=N[keep],C[keep],L[keep]; D=C/(6*N)
 grid=[]; best=None
 for A,B,a,b in product([100,500,2000],[100,1000,5000],[.2,.35,.5],[.2,.35,.5]):
  result=fit(np.array([np.log(1.8),np.log(A),np.log(B),a,b]),N,D,L); grid.append([result.fun,bool(result.success)])
  if best is None or result.fun<best.fun: best=result
 p=unpack(best.x); print('fit',p,best.fun,flush=True); rng=np.random.default_rng(1806); boots=[]; status=[]
 for i in range(400):
  idx=rng.integers(0,len(N),len(N)); res=fit(best.x,N[idx],D[idx],L[idx]); boots.append(unpack(res.x)); status.append(bool(res.success))
  if i%50==0: print('bootstrap',i,flush=True)
 boots=np.array(boots); printed=np.array([1.69,406.4,410.7,.34,.28]); budgets=np.logspace(18,25,100); opt=np.array(optimum(p,budgets)); ref=np.array(optimum(printed,budgets)); nc,dc=optimum(p,5.76e23); ratios=np.array([optimum(q,5.76e23)[1]/optimum(q,5.76e23)[0] for q in boots]); residual=np.log(law(p,N,D))-np.log(L); residual_printed=np.log(law(printed,N,D))-np.log(L)
 save('scaling',{'parameter_order':['E','A','B','alpha','beta'],'fit':p,'standard_errors':boots.std(0,ddof=1),'original_rows':len(rows),'excluded_rows':removed,'fitted_rows':len(N),'grid_starts':len(grid),'best_success':bool(best.success),'bootstrap_count':400,'bootstrap_successes':sum(status),'seed':1806,'mean_log_residual':residual.mean(),'printed_mean_log_residual':residual_printed.mean(),'chinchilla_budget':5.76e23,'N_star':nc,'D_star':dc,'D_over_N':dc/nc,'D_over_N_95CI':np.quantile(ratios,[.025,.975]),'printed_D_over_N':float(optimum(printed,5.76e23)[1]/optimum(printed,5.76e23)[0]),'compute':budgets,'optimal_N':opt[0],'optimal_D':opt[1],'seconds':time.monotonic()-start})
 np.savez(ART/'scaling.npz',N=N,D=D,C=C,L=L,bootstrap=boots,grid=grid,theta=best.x,residual=residual,residual_printed=residual_printed)
 plt.figure(figsize=(10,4)); ax=plt.subplot(121); ax.scatter(N,L,c='lightgray',s=12)
 ns=np.logspace(7,11,200)
 for c in [1e19,1e20,1e21]:
  line,=ax.plot(ns,law(p,ns,c/(6*ns)),label=f'C={c:.0e}'); mask=abs(C/c-1)<.15; ax.scatter(N[mask],L[mask],color=line.get_color(),s=15)
 ax.set(xscale='log',xlabel='parameters N',ylabel='loss',ylim=(1.8,5.5)); ax.legend(); ax=plt.subplot(122); im=ax.scatter(L,law(p,N,D),c=np.log10(C),s=15); ax.plot([2,5],[2,5],'k--'); ax.set(xlabel='measured loss',ylabel='fitted loss'); plt.colorbar(im,ax=ax,label='log10 FLOP'); fig('scaling_fit')
 plt.figure(figsize=(10,4)); ax=plt.subplot(121); ax.scatter(N,residual,s=10,label='refit'); ax.scatter(N,residual_printed,s=10,label='printed'); ax.axhline(0,c='k',ls=':'); ax.set(xscale='log',xlabel='parameters N',ylabel='predicted - observed log loss'); ax.legend(); ax=plt.subplot(122); ax.hist(residual,bins=25,alpha=.6,label='refit'); ax.hist(residual_printed,bins=25,alpha=.6,label='printed'); ax.set_xlabel('log residual'); ax.legend(); fig('scaling_residuals')
 plt.figure(figsize=(7,4)); plt.loglog(budgets,opt[0],label='N* refit'); plt.loglog(budgets,opt[1],label='D* refit'); plt.loglog(budgets,ref[0],'--',label='N* printed'); plt.loglog(budgets,ref[1],'--',label='D* printed'); plt.scatter([5.76e23]*2,[70e9,1.4e12],marker='*',s=100,c='black',label='Chinchilla'); plt.xlabel('training compute (FLOP)'); plt.ylabel('parameters / tokens'); plt.legend(); fig('scaling_optimum')
if __name__=='__main__': main()
