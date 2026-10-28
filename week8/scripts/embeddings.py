from common import *
from safetensors.numpy import load_file
from tokenizers import Tokenizer
raw=load_file(str(DATA/'model.safetensors')); print({k:v.shape for k,v in raw.items()})
e=next(v for v in raw.values() if v.shape==(29528,256)).astype(np.float64)
e/=np.linalg.norm(e,axis=1,keepdims=True)
tok=Tokenizer.from_file(str(DATA/'tokenizer.json'))
relations={'country-capital':'iran tehran turkey ankara russia moscow china beijing france paris sweden stockholm japan tokyo peru lima kenya nairobi','male-female':'king queen man woman he she brother sister father mother son daughter uncle aunt boy girl husband wife','comparative':'big bigger small smaller fast faster dark darker cold colder old older strong stronger long longer tall taller'}
control=e[np.random.default_rng(1730).permutation(len(e))]; result={}
plt.figure(figsize=(13,4))
for j,(name,words) in enumerate(relations.items()):
    words=words.split(); ids=[tok.token_to_id(w) for w in words]; assert all(i is not None for i in ids)
    ax=plt.subplot(1,3,j+1); result[name]={'pairs':list(zip(words[::2],words[1::2])),'token_ids':ids}
    for table,label,sign,color in [(e,'trained',1,'#247ba0'),(control,'shuffled',-1,'#888888')]:
        offsets=table[ids[1::2]]-table[ids[::2]]; others=(offsets.sum(0)-offsets)/8
        c=np.sum(offsets*others,axis=1)/(np.linalg.norm(offsets,axis=1)*np.linalg.norm(others,axis=1))
        result[name][label]={'cosines':c,'mean':c.mean()}
        for x in c: ax.arrow(0,0,x,sign*np.sqrt(1-x*x),width=.004,head_width=.035,color=color,alpha=.7,length_includes_head=True)
    ax.axvline(0,color='black',ls=':'); ax.axhline(0,color='black'); ax.set(xlim=(-.5,1.1),ylim=(-1.1,1.1),title=name,xlabel='leave-one-out cosine',aspect='equal')
fig('embeddings'); save('embeddings',result); print(result)
