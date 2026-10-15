#!/usr/bin/env python3
"""Compare a running native build with public mistral3-reference.py fixtures."""
import argparse,base64,json,os,subprocess,time,urllib.request,urllib.error
from pathlib import Path
import numpy as np
from safetensors import safe_open
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fixtures',type=Path,required=True)
parser.add_argument('--binary',type=Path,required=True)
parser.add_argument('--model')
parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
parser.add_argument('--dtype',choices=['float32','bfloat16','float16'],default='float32')
parser.add_argument('--pooling',choices=['last-token','mean','auto'],default='last-token')
parser.add_argument('--port',type=int,default=18781)
parser.add_argument('--max-batch-tokens',type=int,default=2048)
args=parser.parse_args()
fixtures=args.fixtures.resolve()
root=fixtures
model=args.model or str(fixtures/'model')
device=args.device
dtype=args.dtype
port=args.port
log=(root/('http-'+device+'-'+fixtures.name+'.log')).open('w')
env=dict(os.environ)
env.setdefault('OMP_NUM_THREADS','1')
env.setdefault('MKL_NUM_THREADS','1')
command=[str(args.binary.resolve()),'--model-id',model,'--backend','libtorch','--torch-device',device,'--dtype',dtype,'--hostname','127.0.0.1','--port',str(port),'--max-batch-tokens',str(args.max_batch_tokens),'--auto-truncate','--tokenization-workers','1']
if args.pooling!='auto':command.extend(['--pooling',args.pooling])
proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
def req(path,payload=None):
 r=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(r,timeout=120) as response:return json.load(response)
 except urllib.error.HTTPError as e:raise RuntimeError(str(e.code)+' '+e.read().decode())
results=[]
try:
 for _ in range(240):
  if proc.poll() is not None:raise RuntimeError('Server exited; inspect '+log.name)
  try:req('/info');break
  except (urllib.error.URLError,ConnectionError):time.sleep(.5)
 else:raise RuntimeError('Startup timeout')
 for c in json.loads((fixtures/'cases.json').read_text()):
  parts=[]
  for i in range(c['images']):
   encoded=base64.b64encode((fixtures/f'{c["name"]}-{i}.png').read_bytes()).decode()
   parts.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+encoded}})
  parts.append({'type':'text','text':'Describe the shapes and colors.'})
  messages=[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':parts}]
  actual=np.asarray(req('/embed_all',{'inputs':messages})[0],dtype=np.float32)
  with safe_open(fixtures/f'{c["name"]}.safetensors',framework='np') as f:
   expected=f.get_tensor('hidden')
   pooled=f.get_tensor('pooled') if 'pooled' in f.keys() else expected[-1]
  assert actual.shape==expected.shape,(c['name'],actual.shape,expected.shape)
  delta=float(np.max(np.abs(actual-expected)))
  a64,b64=actual.astype(np.float64),expected.astype(np.float64)
  cosine=float(np.clip(np.sum(a64*b64)/(np.linalg.norm(a64)*np.linalg.norm(b64)),-1,1))
  assert cosine>(.999 if dtype=='bfloat16' else .99999),(c['name'],delta,cosine)
  relative=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
  assert (relative<.05 if dtype=='bfloat16' else delta<.002),(c['name'],delta,cosine,relative)
  # Pooling and normalization agree with the same reference, through a second API path.
  embed=np.asarray(req('/v1/embeddings',{'model':model,'input':messages,'encoding_format':'float'})['data'][0]['embedding'])
  wanted=pooled;wanted=wanted/np.linalg.norm(wanted)
  assert np.isclose(np.linalg.norm(embed),1.0,rtol=1e-5,atol=1e-6),(c['name'],np.linalg.norm(embed))
  assert np.dot(embed,wanted)/(np.linalg.norm(embed)*np.linalg.norm(wanted))>(.999 if dtype=='bfloat16' else .99999)
  last_cosine=float(np.dot(embed,wanted)/(np.linalg.norm(embed)*np.linalg.norm(wanted)))
  results.append({'case':c['name'],'tokens':actual.shape[0],'max_absolute_difference':delta,'cosine':cosine,'relative_l2_error':relative,'pooled_cosine':float(np.clip(last_cosine,-1,1))})
  print(results[-1],flush=True)
 # Reject truncation that would leave a partial image or silently discard a modality.
 try:req('/embed',{'inputs':[{'role':'user','content':[{'type':'text','text':'[IMG]'}]}]})
 except RuntimeError as e:assert 'reserved' in str(e),str(e)
 else:raise AssertionError('Reserved image marker was accepted')
finally:
 proc.terminate()
 try:proc.wait(timeout=15)
 except subprocess.TimeoutExpired:proc.kill();proc.wait()
 log.close()
(root/('http-'+device+'-'+fixtures.name+'-results.json')).write_text(json.dumps(results,indent=2)+'\n')
