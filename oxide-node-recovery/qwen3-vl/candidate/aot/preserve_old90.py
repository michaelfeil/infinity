#!/usr/bin/env python3
"""Compare every published old 90 export's ELF .text and driver resources with
the new 96-export Qwen3-VL device bundle on SM90/SM100/SM120. Source-only check; no GPU."""
import argparse, hashlib, json, os, re, struct, subprocess
from pathlib import Path

def sections(path):
    data=path.read_bytes()
    if data[:6]!=b"\x7fELF\x02\x01": raise RuntimeError(f"expected ELF64 {path}")
    offset=struct.unpack_from("<Q",data,40)[0]; size,count,strings=struct.unpack_from("<HHH",data,58)
    headers=[struct.unpack_from("<IIQQQQIIQQ",data,offset+i*size) for i in range(count)]
    table=headers[strings]; names=data[table[4]:table[4]+table[5]]
    result={}
    for h in headers:
        start=h[0]; end=names.index(0,start); name=names[start:end].decode()
        if name.startswith(".text."): result[name[6:]]=data[h[4]:h[4]+h[5]]
    return result

def resources(path,cuobjdump):
    text=subprocess.check_output([cuobjdump,"--dump-resource-usage",str(path)],text=True)
    return dict((m.group(1),m.group(2).strip()) for m in re.finditer(r"Function\s+(\w+):\s*\n\s*([^\n]+)",text))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--old',type=Path,required=True); p.add_argument('--new',type=Path,required=True); p.add_argument('--expected-old',type=Path,required=True); p.add_argument('--expected-new',type=Path,required=True); p.add_argument('--cuobjdump',default=os.environ.get('CUOBJDUMP','/usr/local/cuda-13.1/bin/cuobjdump')); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    old_names=set(a.expected_old.read_text().split()); new_names=set(a.expected_new.read_text().split())
    old_manifest=json.loads((a.old/'manifest.json').read_text()); new_manifest=json.loads((a.new/'manifest.json').read_text())
    failures=[]
    if set(old_manifest['kernels'])!=old_names: failures.append({'check':'old manifest','detail':'old bundle symbol set differs from expected old90'})
    if set(new_manifest['kernels'])!=new_names: failures.append({'check':'new manifest','detail':'new bundle symbol set differs from expected new96'})
    checks=[]
    for arch in ('sm_90','sm_100','sm_120'):
        a_sec=sections(a.old/arch/'kernels.cubin'); b_sec=sections(a.new/arch/'kernels.cubin')
        a_res=resources(a.old/arch/'kernels.cubin',a.cuobjdump); b_res=resources(a.new/arch/'kernels.cubin',a.cuobjdump)
        for name in sorted(old_names):
            ok=a_sec.get(name)==b_sec.get(name) and a_res.get(name)==b_res.get(name)
            checks.append({'architecture':arch,'symbol':name,'text_identical':a_sec.get(name)==b_sec.get(name),'resources_identical':a_res.get(name)==b_res.get(name),'old_text_sha256':hashlib.sha256(a_sec.get(name,b'')).hexdigest(),'new_text_sha256':hashlib.sha256(b_sec.get(name,b'')).hexdigest(),'old_resources':a_res.get(name),'new_resources':b_res.get(name)})
            if not ok: failures.append({'check':f'{arch}:{name}','detail':'old semantic change'})
    additive=sorted(new_names-old_names)
    if additive!=['infinity_decoder_rows_bf16','infinity_qwen3vl_add_bf16','infinity_qwen3vl_gelu_tanh_bf16','infinity_qwen3vl_pos_interp','infinity_qwen3vl_rope2d_bf16','infinity_qwen3vl_silu_bf16']:
        failures.append({'check':'additive exports','detail':additive})
    result={'schema':'oxide-qwen3-vl-e582-old90-preservation/v1','status':'passed' if not failures else 'failed','old_bundle':str(a.old),'new_bundle':str(a.new),'old_exports':len(old_names),'new_exports':len(new_names),'additive':additive,'checks':checks,'failures':failures,'scope':'machine/resource preservation only; no GPU execution, math or performance claim'}
    a.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({'status':result['status'],'checks':len(checks),'failures':len(failures),'additive':additive}))
    if failures: raise SystemExit(1)

if __name__=='__main__':main()
