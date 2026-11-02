#!/usr/bin/env python3
"""Copy-only all3 union130 and actual125 per-architecture ELF preservation."""
from pathlib import Path
import hashlib,json,shutil,sys
sys.dont_write_bytecode=True
sys.path.insert(0,'/tmp/oxide-device130-compose-source-v1/evidence')
from elf_tools import sections,resources
from ptx_symbol_review import verify
R=Path(__file__).resolve().parent
H100=Path('/tmp/oxide-device130-compose-source-v1/kernels')
OTHER=R/'compiled-sm100-sm120';OUT=R/'kernels'
BASE=Path('/tmp/infinity-oxide-parity-20261101/device-union124-all3-v1/kernels')
NORMS=Path('/tmp/infinity-oxide-parity-20261101/qwen35-norm-vector64-v1/kernels')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    m90=json.load(open(H100/'manifest.json'));mo=json.load(open(OTHER/'manifest.json'))
    assert m90['kernels']==mo['kernels'] and m90['sources']==mo['sources']
    assert len(m90['kernels'])==130
    assert not OUT.exists();OUT.mkdir()
    for arch,root in [('sm_90',H100),('sm_100',OTHER),('sm_120',OTHER)]:shutil.copytree(root/arch,OUT/arch)
    for p in H100.iterdir():
        if p.is_file() and p.name!='manifest.json':shutil.copy2(p,OUT/p.name)
    merged=dict(m90);merged['architectures']=m90['architectures']+mo['architectures'];merged['requested_architectures']=['sm_90','sm_100','sm_120'];merged['complete_target_set']=True
    merged['composition_provenance']={'operation':'copy-only manifest union','SM90_qualified_math':'unchanged original artifact, actualRootH5120 Native20 plain/mem/init','SM100_SM120':'CPU compile only; no runtime claim','input_manifests':{str(H100/'manifest.json'):sha(H100/'manifest.json'),str(OTHER/'manifest.json'):sha(OTHER/'manifest.json')}}
    (OUT/'manifest.json').write_text(json.dumps(merged,indent=2)+'\n')
    base=json.load(open(BASE/'manifest.json'));norm=json.load(open(NORMS/'manifest.json'))
    old=set(base['kernels'])-{'infinity_qwen35vl_pos_interp'}
    nset={'infinity_qwen35_norm_vector_memory_bf16','infinity_qwen35_residual_norm_vector_memory_bf16'}
    assert len(old)==123 and nset<=set(norm['kernels']) and not nset&old
    checks=[];ptx={}
    for arch in ['sm_90','sm_100','sm_120']:
        new=sections(OUT/arch/'kernels.cubin');nr=resources(OUT/arch/'kernels.cubin')
        for donor,names in [(BASE,old),(NORMS,nset)]:
            prev=sections(donor/arch/'kernels.cubin');pr=resources(donor/arch/'kernels.cubin')
            for name in sorted(names):checks.append({'architecture':arch,'symbol':name,'donor':str(donor),'text_control_bytes_identical':prev[name]==new[name],'resources_identical':pr[name]==nr[name],'old_sha256':hashlib.sha256(prev[name]).hexdigest(),'new_sha256':hashlib.sha256(new[name]).hexdigest(),'old_resources':pr[name],'new_resources':nr[name]})
        pt=verify((BASE/arch/'kernels.ptx').read_text(),(OUT/arch/'kernels.ptx').read_text())
        (R/'evidence'/f'{arch}-old123-ptx.json').write_text(json.dumps(pt,indent=2)+'\n');ptx[arch]=pt['status']
    failures=[c for c in checks if not c['text_control_bytes_identical'] or not c['resources_identical']]
    report={'schema':'central130-all3-machine-proof/v1','status':'PASS' if not failures and all(v=='PASS' for v in ptx.values()) else 'FAIL','checks':checks,'failures':failures,'check_count':len(checks),'PTX':ptx,'old125_per_architecture':True,'intentional_old_change':['infinity_qwen35vl_pos_interp'],'SM90_copy_sha_identical':sha(H100/'sm_90/kernels.cubin')==sha(OUT/'sm_90/kernels.cubin'),'GPU_by_agent':False,'SM100_SM120_runtime_qualified':False}
    (R/'evidence/old125-all3-preservation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2));assert report['status']=='PASS'
if __name__=='__main__':main()
