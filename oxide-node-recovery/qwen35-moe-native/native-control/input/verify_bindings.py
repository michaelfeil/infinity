"""CPU-only check of actual CUDA BLAS relocation targets in Torch and the probe."""
import ctypes as C
import json
from pathlib import Path
import subprocess
import torch

class DlInfo(C.Structure):
    _fields_=[('name',C.c_char_p),('base',C.c_void_p),('symbol',C.c_char_p),('address',C.c_void_p)]

def verify(probe_path):
    maps=Path('/proc/self/maps').read_text().splitlines()
    dl=C.CDLL('libdl.so.2');dl.dladdr.argtypes=[C.c_void_p,C.POINTER(DlInfo)];dl.dladdr.restype=C.c_int
    objects=[Path(torch.__file__).resolve().parent/'lib/libtorch_cuda.so',Path(probe_path)]
    result={'gpu_initialized':torch.cuda.is_initialized(),'objects':{},'all_blas_relocations_host_vendor':True}
    for obj in objects:
        matched=[line.split() for line in maps if line.split()[-1]==str(obj)]
        if not matched:raise RuntimeError(f'Object not mapped: {obj}')
        # Torch's last LOAD segment has p_vaddr != p_offset; do not infer
        # load bias by minimizing start-file_offset across all mappings.
        # Both inspected objects have their first LOAD at vaddr/file offset0.
        starts=[int(parts[0].split('-')[0],16) for parts in matched if int(parts[2],16)==0]
        if len(starts)!=1:raise RuntimeError(f'Expected one zero-offset LOAD mapping: {obj}')
        base=starts[0]
        targets=[]
        process=subprocess.Popen(['readelf','-rW',str(obj)],stdout=subprocess.PIPE,text=True)
        for line in process.stdout:
            parts=line.split()
            if len(parts)<5 or not parts[2].startswith('R_X86_64_'):continue
            symbol=parts[4].split('@')[0]
            if not symbol.startswith('cublas'):continue
            if parts[2] not in ['R_X86_64_JUMP_SLOT','R_X86_64_GLOB_DAT']:continue
            pointer=C.c_void_p.from_address(base+int(parts[0],16)).value
            info=DlInfo();found=dl.dladdr(C.c_void_p(pointer),C.byref(info))
            path=info.name.decode() if found and info.name else None
            good=path is not None and path.startswith('/tmp/oxide-host-cublas-130200/')
            targets.append({'symbol':symbol,'resolved_path':path,'host_vendor':good})
            result['all_blas_relocations_host_vendor'] &= good
        if process.wait()!=0:raise RuntimeError('readelf failed')
        if not targets:raise RuntimeError(f'No BLAS relocation targets found: {obj}')
        result['objects'][str(obj)]=targets
    version=C.CDLL(str(probe_path));version.cublasLtGetVersion.restype=C.c_size_t
    result['probe_dependency_cublaslt_version']=version.cublasLtGetVersion()
    return result

if __name__=='__main__':
    p=Path(__file__).resolve().parent;torch.ops.load_library(str(p/'gemm_probe.so'))
    result=verify(p/'gemm_probe.so');(p/'host-vendor-binding-proof.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'gpu_initialized':result['gpu_initialized'],'all_blas_relocations_host_vendor':result['all_blas_relocations_host_vendor'],'version':result['probe_dependency_cublaslt_version'],'relocation_counts':{k:len(v) for k,v in result['objects'].items()}}))
