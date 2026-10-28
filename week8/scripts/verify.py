"""Keep implementation-test success separate from handout target mismatches."""
from common import *
import hashlib,platform,subprocess,sys,importlib.metadata

def main():
 proc=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'scripts'),'-p','test_week8.py','-v'],capture_output=True,text=True)
 (ART/'tests.log').write_text(proc.stdout+proc.stderr)
 read=lambda n:json.loads((EVI/f'{n}.json').read_text())
 e,c,m,p,r,t,g,s=[read(n) for n in ['embeddings','counts','memory','pictures','recall','transformer','generation','scaling']]; ml=read('mlp'); corpus_meta=read('corpus'); checks={}
 for key,v in e.items(): checks['embedding_'+key]=v['trained']['mean']>=.3 and abs(v['shuffled']['mean'])<=.15
 checks['story_split']=corpus_meta['stories']==[24867,2763]; checks['tokenizer_roundtrip']=corpus_meta['val']['roundtrip_failures']==3
 checks['bigram_normalization']=c['verification']['max_normalization_error']<1e-12; checks['bigram_sampling']=c['verification']['bigram_chi2']['p']>.001
 checks['count_optimum_n3_or4']=min(range(1,7),key=lambda n:c[str(n)]['bpc']) in [3,4]
 ref_nats=[6.238,5.230,3.152,2.254]; ref_ppl=[512,186.8,23.4,9.53]; ref_bpc=[3.747,3.141,1.893,1.354]
 for i,k in enumerate(['uniform','1','2','3']):
  checks[f'reference_{k}_nats']=abs(c[k]['nats']-ref_nats[i])<.03; checks[f'reference_{k}_ppl']=abs(c[k]['ppl']/ref_ppl[i]-1)<.03; checks[f'reference_{k}_bpc']=abs(c[k]['bpc']-ref_bpc[i])<.03
 for n,v in zip(range(3,7),[.017,.086,.211,.357]): checks[f'reference_{n}_unseen']=abs(c[str(n)]['unseen']-v)<.02
 for n,v in [(5,1.56),(6,1.93)]: checks[f'reference_{n}_bpc']=abs(c[str(n)]['bpc']-v)<.03
 checks['memory_outer_capacity']=50<=m['capacities'][2][0]<=85; checks['memory_least_squares_capacity']=100<=m['capacities'][2][1]<=170; checks['memory_exact_recall']=m['least_squares_max_R_M_le_d']<1e-10
 for i in range(3): checks[f'memory_slope_{i}']=abs(m['capacity_slopes'][i]-1)<=.1
 checks['memory_slope_2power']=abs(m['capacity_slopes'][3]-2)<=.15; checks['memory_slope_3power']=m['capacity_slopes'][4]>2.5
 checks['mlp_matches_actual_bigram']=abs(ml['nats']-c['2']['nats'])<.01
 pic=np.array(p['ratios_median_20']); checks['picture_outer_prediction']=abs(pic[2,0]/4-1)<.15; checks['picture_relu_prediction']=abs(pic[2,1]/np.sqrt(8)-1)<.15; checks['picture_relu2_failure']=pic[3,2]>1; checks['picture_softmax_success']=bool(np.all(pic[:,3]<.1))
 checks['attention_all_distances']=min(r['accuracy_by_distance'].values())>.95
 checks['transformer_beats_counts']=t['bpc']<min(c[str(n)]['bpc'] for n in range(1,7)); checks['transformer_reference_threshold']=t['bpc']<1.315; checks['transformer_32plus_context']=max(t['position_loss'][31:])<c['3']['nats']
 for k,v in g['sampler_tests'].items(): checks['sampler_'+k]=v if isinstance(v,bool) else v['p']>.001
 checks['self_consistency_T1']=abs(g['consistency']['1.0']['z'])<4; checks['self_consistency_T07']=abs(g['consistency']['0.7']['z'])>40
 E,A,B,a,b=s['fit']; checks['scaling_parameters']=.32<=a<=.38 and .32<=b<=.42 and 1.75<=E<=1.90; checks['scaling_residuals']=abs(s['mean_log_residual'])<.005 and s['printed_mean_log_residual']>.01; checks['scaling_compute_split']=10<s['D_over_N']<30; checks['bootstrap_400_successes']=s['bootstrap_successes']==400
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(DATA.iterdir())+list(EVI.glob('*.png'))+[ROOT/'week8-learning-sheet.pdf'] if p.is_file()}
 save('verification',{'implementation_tests_passed':proc.returncode==0,'implementation_test_log':proc.stdout+proc.stderr,'target_checks':checks,'failed_targets':[k for k,v in checks.items() if not v],'all_handout_targets_met':all(checks.values()),'versions':{k:importlib.metadata.version(k) for k in ['jax','jaxlib','optax','numpy','scipy','sentencepiece','tokenizers','safetensors']},'python':platform.python_version(),'platform':platform.platform(),'sha256':hashes})
 print(proc.stdout+proc.stderr); print('Unmet handout targets:',[k for k,v in checks.items() if not v]); return proc.returncode
if __name__=='__main__': sys.exit(main())
