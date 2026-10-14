"""Independent trained encoder parity; Python is used only by this test.

Usage: python check_pretrained_encoder_reference.py CHECKPOINT NATIVE_FFI OUTPUT_JSON
Build encoder_reference_ffi.cpp against the native backend first. Native uses
CUDA2; Transformers uses CUDA3. Both consume exact sequence lengths without padding.
The pinned checkpoint must contain config.json, model.safetensors and revision.json.
"""
import ctypes as c,json,pathlib,sys,time
import numpy as np,torch
from safetensors.numpy import load_file
from transformers import AutoModel,AutoModelForSequenceClassification
p=pathlib.Path(sys.argv[1]);cfg=json.loads((p/'config.json').read_text());lib=c.CDLL(sys.argv[2])
lib.fixture_create.restype=c.c_void_p;lib.fixture_error.restype=c.c_char_p
lib.fixture_option.argtypes=[c.c_void_p,c.c_char_p,c.c_char_p];lib.fixture_weight.argtypes=[c.c_void_p,c.c_char_p,c.c_void_p,c.c_void_p,c.c_int64]
lib.fixture_ready.argtypes=[c.c_void_p,c.c_int];lib.fixture_forward.argtypes=[c.c_void_p]+[c.c_void_p]*4+[c.c_int64]*2+[c.c_int]*2+[c.c_void_p];lib.fixture_delete.argtypes=[c.c_void_p]
def check(s):
 if s:raise RuntimeError(lib.fixture_error().decode())
def flatten(o,pre=''):
 for k,v in o.items():
  if isinstance(v,dict):yield from flatten(v,pre+k+'.')
  elif v is not None:yield pre+k,v if isinstance(v,str) else json.dumps(v)
def cosine(a,b):
 a=a.astype(np.float64);b=b.astype(np.float64);return np.sum(a*b,-1)/np.sqrt(np.sum(a*a,-1)*np.sum(b*b,-1))
handle=lib.fixture_create()
for k,v in flatten(cfg):lib.fixture_option(handle,k.encode(),v.encode())
for k,v in load_file(p/'model.safetensors').items():
 v=v.astype(np.float32);shape=np.array(v.shape,dtype=np.int64);check(lib.fixture_weight(handle,k.encode(),v.ctypes.data,shape.ctypes.data,v.ndim))
check(lib.fixture_ready(handle,1));torch.set_num_threads(1)
classification=cfg['model_type']=='deberta-v2';model=(AutoModelForSequenceClassification if classification else AutoModel).from_pretrained(p,local_files_only=True,dtype=torch.float16,attn_implementation='eager').eval().to('cuda:3')
backbone=model.deberta if classification else model
records=[]
for lengths in [[32],[128]*8,[32+32*(i%8) for i in range(32)],[512]*32]:
 cls,sep=(1,2) if classification else (0,2)
 ids=np.array([v for r,n in enumerate(lengths) for v in [cls]+[1000+(i*17+r*31)%20000 for i in range(1,n-1)]+[sep]],dtype=np.int64)
 types=np.zeros_like(ids);pos=np.array([i for n in lengths for i in range(n)],dtype=np.int64);offset=np.array([0]+list(np.cumsum(lengths)),dtype=np.int32)
 actual=np.zeros((len(ids),cfg['hidden_size']),dtype=np.float32)
 args=[handle,ids.ctypes.data,types.ctypes.data,pos.ctypes.data,offset.ctypes.data,len(lengths),max(lengths),1]
 check(lib.fixture_forward(*args,0,actual.ctypes.data))
 expected=[];expected_logits=[]
 with torch.inference_mode():
  for r,n in enumerate(lengths):
   x=torch.tensor(ids[offset[r]:offset[r+1]],device='cuda:3')[None]
   expected.append(backbone(x).last_hidden_state[0].float().cpu().numpy())
   if classification:expected_logits.append(model(x).logits[0].float().cpu().numpy())
 expected=np.concatenate(expected)
 rowcos=[];clscos=[]
 for r,n in enumerate(lengths):
  a,b=actual[offset[r]:offset[r+1]],expected[offset[r]:offset[r+1]];rowcos.append(float(cosine(a.mean(0),b.mean(0))));clscos.append(float(cosine(a[0],b[0])))
 report=dict(lengths=lengths,actual_tokens=sum(lengths),mean_cosine=min(rowcos),cls_cosine=min(clscos),token_cosine=float(cosine(actual,expected).min()),max_absolute=float(np.max(np.abs(actual-expected))))
 if classification:
  logits=np.zeros((len(lengths),len(cfg['id2label'])),dtype=np.float32);check(lib.fixture_forward(*args,1,logits.ctypes.data));ref=np.array(expected_logits);report['classifier_cosine']=float(cosine(logits,ref).min());report['classifier_max_absolute']=float(np.max(np.abs(logits-ref)))
 records.append(report);print(json.dumps(report),flush=True)
 assert min(report['mean_cosine'], report['cls_cosine'], report['token_cosine'])>=.999,report
 if classification:assert report['classifier_cosine']>=.999,report
lib.fixture_delete(handle)
pathlib.Path(sys.argv[3]).write_text(json.dumps(dict(checkpoint=json.loads((p/'revision.json').read_text()),dtype='float16',reference_torch_version=torch.__version__,reference_transformers_version=__import__('transformers').__version__,torch_gpu=2,reference_gpu=3,token_padding=False,cosine_gate=.999,reference='Transformers eager; independent exact-length calls; FP16 GPU3 versus native packed GPU2',results=records),indent=2)+'\n')
