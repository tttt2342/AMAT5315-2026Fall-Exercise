from common import *
import sentencepiece as spm
sp=spm.SentencePieceProcessor(model_file=str(DATA/'tok512.model'))
stories=[s.strip() for s in (DATA/'TinyStoriesV2-GPT4-valid.txt').read_text().split('<|endoftext|>') if s.strip()]
n=int(.9*len(stories)); arrays={}; meta={'stories':[n,len(stories)-n],'vocab':sp.vocab_size(),'eos':sp.eos_id(),'source':'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-valid.txt'}
for name,split in [('train',stories[:n]),('val',stories[n:])]:
    encoded=sp.encode(split); tokens=np.array([t for ids in encoded for t in ids+[sp.eos_id()]],dtype=np.int32)
    arrays[name]=tokens; meta[name]={'characters':sum(map(len,split)),'tokens':len(tokens),'characters_per_token':sum(map(len,split))/len(tokens),'roundtrip_failures':sum(sp.decode(ids)!=s for ids,s in zip(encoded,split))}
np.savez(ART/'tokens.npz',**arrays); save('corpus',meta); print(meta,flush=True)
