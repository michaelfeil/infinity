#!/usr/bin/env python3
"""Balanced N,O,O,N paired LibTorch vs public Oxide timing controller."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

PACKET = Path(__file__).resolve().parents[1]
PUBLIC = Path('/tmp/infinity-plain-cuda')
EXPECTED_PUBLIC = '252789b05ea2e09c14971c04dfcce458f863fead'
MODEL = Path('/tmp/infinity-oxide-bench/models/Qwen--Qwen3-30B-A3B-ad44e777')
NATIVE_ROOT = Path('/tmp/infinity-oxide-parity-20261101/qwen3-moe-native-capi-v1')
NATIVE_SDK = Path('/tmp/native214-fa4-mode20/install/lib/libtei_torch.so')
NATIVE_SDK_SHA = 'b66d70b61b39e1bf0d565aca33bac7c5e21849677396ce5216f5baf73027d37e'
NATIVE_HELPER = NATIVE_ROOT / 'bench_helper.so'
NATIVE_HELPER_SHA = '679603a8b729a3aa4d625ad3ea23fa090e90842c9bded30def3ba2c522cd2ddb'
NATIVE_HEADER = NATIVE_ROOT / 'bridge.h'
NATIVE_HEADER_SHA = 'ed5cd44f735cbe21160f43fbdec0f4e003acd477497ab4141e2625c37c5b3b69'
SOURCE_PACKET = Path('/tmp/infinity-oxide-parity-20261101/qwen3-moe-full45-source-gate-v3')
ROOT_REPORT = Path('/tmp/infinity-oxide-parity-20261101/qwen3-moe-root-full45-source-gate-v3')
NATIVE_REPORT = NATIVE_ROOT / 'root-native-quality-all-v1/report.json'
CASES = PACKET / 'oxide-perf/cases/cases.json'
GOLD_CASES = {'t7': 'ragged7-pooled', 'ragged129': 'ragged129-mixed'}
POOL = 'mean'
WINDOWS = ('native-1', 'oxide-1', 'oxide-2', 'native-2')
NATIVE_KIND = {'native-1', 'native-2'}
OXIDE_KIND = {'oxide-1', 'oxide-2'}

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text())
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def check(c,m):
    if not c: raise RuntimeError(m)

def git_head(path):
    return subprocess.run(['git','-C',str(path),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()

def cases_without_stable(path,out):
    doc=read(path); doc['cases']=[c for c in doc['cases'] if c['id']!='stable2k']
    p=out/'cases-no-stable2k.json'; p.write_text(json.dumps(doc,indent=2)+'\n'); return p

def preflight(args):
    check(git_head(PUBLIC)==EXPECTED_PUBLIC, f'public commit differs: {git_head(PUBLIC)}')
    check(sha(MODEL/'manifest.json')=='55be60f3bdfa41310a73bde32d2fdd352a3f11c6e4f3c2cc4affa2f16550fc88','model manifest differs')
    check(sha(MODEL/'config.json')=='2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297','model config differs')
    cases=read(args.cases); ids=[c['id'] for c in cases['cases']]
    check(ids[:2]==['t7','ragged129'],'T7/ragged129 must be first')
    rows=[]
    for c in cases['cases']:
        p=args.cases.parent/c['batch']
        rows.append({'id':c['id'],'batch':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
    missing=[]
    for label,p,expect in [('native_sdk',args.native_sdk,NATIVE_SDK_SHA),('native_helper',args.native_helper,NATIVE_HELPER_SHA),('native_header',NATIVE_HEADER,NATIVE_HEADER_SHA)]:
        if not Path(p).is_file(): missing.append({'input':label,'path':str(p),'expected_sha256':expect})
        else: check(sha(p)==expect,f'{label} hash differs')
    check(sha(NATIVE_REPORT)=='17446c7c4db65b0c58a4a5bc0387102f1c3b165fe81480d2cb407249070fdb83','native gold report hash differs')
    native=read(NATIVE_REPORT); row=next(r for r in native['pool_reports'] if r['pool']==POOL)
    gold={}
    for case_id, native_id in GOLD_CASES.items():
        c=next(x for x in row['checks'] if x['id']==native_id)
        check(sha(c['native_report'])==c['sha256'],f'gold case hash differs: {native_id}')
        gold[case_id]={'native_report':c['native_report'],'sha256':c['sha256'],'batch_sha256':c['batch_sha256']}
    if args.cpu_only:
        return {'schema':'qwen3-moe-paired-perf-preflight/v1','status':'cpu-prepared-with-missing-native-sdk' if missing else 'cpu-prepared','public_commit':EXPECTED_PUBLIC,'model':{'manifest_sha256':sha(MODEL/'manifest.json'),'config_sha256':sha(MODEL/'config.json')},'cases':rows,'gold':gold,'missing_inputs':missing,'native_sdk':str(args.native_sdk),'native_helper':str(args.native_helper)}
    check(not missing,f'missing native inputs: {missing}')
    check(args.oxide_binary.is_file(),f'oxide binary missing: {args.oxide_binary}')
    check(args.oxide_build.is_file(),f'oxide build pin missing: {args.oxide_build}')
    build=read(args.oxide_build)
    check(build.get('binary_sha256')==sha(args.oxide_binary),'oxide binary hash differs from build pin')
    return {'schema':'qwen3-moe-paired-perf-preflight/v1','status':'ready','public_commit':EXPECTED_PUBLIC,'model':{'manifest_sha256':sha(MODEL/'manifest.json'),'config_sha256':sha(MODEL/'config.json')},'cases':rows,'gold':gold,'missing_inputs':[],'oxide_binary':str(args.oxide_binary),'oxide_binary_sha256':sha(args.oxide_binary),'oxide_build':str(args.oxide_build),'oxide_build_sha256':sha(args.oxide_build),'native_sdk':str(args.native_sdk),'native_helper':str(args.native_helper)}

def native_env(args):
    env=dict(os.environ)
    for k in list(env):
        if k.startswith(('INFINITY_','LD_DEBUG')): env.pop(k)
    preload=[str(Path('/usr/local/cuda-13.1/lib64')/n) for n in ('libcublas.so.13','libcublasLt.so.13')]
    env.update({'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','LD_BIND_NOW':'1','LD_LIBRARY_PATH':'/usr/local/cuda-13.1/lib64:/usr/local/cuda/lib64:'+env.get('LD_LIBRARY_PATH',''),'INFINITY_TORCH_CUDA_GRAPHS':'1','INFINITY_TORCH_CUDA_GRAPH_REUSE':'3','INFINITY_TORCH_CUDA_GRAPH_ENTRIES':'4','INFINITY_TORCH_CUDA_GRAPH_MAX_TOKENS':'4096','INFINITY_TORCH_CUDA_GRAPH_MAX_BYTES':str(512<<20),'INFINITY_TORCH_CUDA_GRAPH_MAX_CAPTURES':'8'})
    if all(Path(p).is_file() for p in preload): env['LD_PRELOAD']=':'.join(preload)
    return env

def oxide_env(args):
    env=dict(os.environ)
    for k in list(env):
        if k.startswith(('INFINITY_','LD_DEBUG')): env.pop(k)
    preload=[str(Path('/usr/local/cuda-13.1/lib64')/n) for n in ('libcublas.so.13','libcublasLt.so.13')]
    env.update({'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','LD_BIND_NOW':'1','LD_LIBRARY_PATH':'/usr/local/cuda-13.1/lib64:/usr/local/cuda/lib64:'+str(args.provider_root/'sm_90')+':'+env.get('LD_LIBRARY_PATH',''),'INFINITY_OXIDE_ATTENTION':'fa4','INFINITY_OXIDE_KERNEL_MODULE':str(args.kernel_root),'INFINITY_FA4_ROOT':str(args.provider_root)})
    if all(Path(p).is_file() for p in preload): env['LD_PRELOAD']=':'.join(preload)
    return env

def run_window(args, name, window_dir, cases_path):
    window_dir.mkdir(parents=False,exist_ok=False)
    is_native=name in NATIVE_KIND
    out=window_dir/('native' if is_native else 'oxide')
    if is_native:
        cmd=[args.native_python,str(PACKET/'native-perf/native_perf_process.py'),'--model',str(args.model),'--pool',args.pool,'--cases',str(cases_path),'--output',str(out),'--warmup',str(args.warmup),'--iterations',str(args.iterations),'--aba-cycles',str(args.aba_cycles),'--library',str(args.native_sdk),'--helper',str(args.native_helper)]
        env=native_env(args)
    else:
        cmd=[str(args.oxide_binary),'--model',str(args.model),'--pool',args.pool,'--cases',str(cases_path),'--output',str(out),'--warmup',str(args.warmup),'--iterations',str(args.iterations),'--aba-cycles',str(args.aba_cycles),'--device','0']
        env=oxide_env(args)
    stdout=window_dir/'stdout.log'; stderr=window_dir/'stderr.log'
    started=time.time()
    with stdout.open('xb') as o, stderr.open('xb') as e:
        proc=subprocess.Popen(cmd,env=env,stdout=o,stderr=e)
        record={'window':name,'backend':'native' if is_native else 'oxide','pid':proc.pid,'parent_pid':os.getpid(),'argv':cmd,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        write(window_dir/'process.json',record)
        rc=proc.wait()
        record.update(returncode=rc,ended_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),seconds=round(time.time()-started,3))
        write(window_dir/'process.json',record)
    check(rc==0,f'{name} failed; see {stderr}')
    summary=read(out/'summary.json')
    check(summary.get('schema') in ('qwen3-moe-native-perf/v1','qwen3-moe-oxide-perf/v1'),'summary schema differs')
    check([c['id'] for c in summary['cases']]==[c['id'] for c in read(cases_path)['cases']],'case order differs')
    for c in summary['cases']:
        check(c['warmup_us'] and len(c['warmup_us'])==args.warmup and len(c['timed_us'])==args.iterations,'warmup/timed count differs')
        check(c['first_call_us']>0 and c['result_sha256'],'first call missing')
    aba=summary.get('aba',[])
    check(len(aba)==args.aba_cycles*2,'ABA count differs')
    by_case={c['id']:c['result_sha256'] for c in summary['cases']}
    check(all(a['result_sha256']==by_case[a['case']] for a in aba),'ABA result differs from steady result')
    if is_native:
        check(summary['graphs']['requested'] is True,'native graphs not requested')
        replay=max([summary['graphs']['after_aba'].get('hits',0)] + [c['graph_stats_after'].get('hits',0) for c in summary['cases']])
        check(replay>0,'native graph replay was not observed')
        proof=summary.get('vendor_binding_proof') or {}
        check(proof.get('gpu_initialized') is False,'native pre-CUDA vendor proof differs')
        check(proof.get('probe_dependency_cublaslt_version')==130200,'vendor Lt version differs')
        expected_vendors={'416a07324bf28948db370fa5c547cb2f7e896ac1bbc0038bc3211ccd49b6e88e','5cd9d979d72807f61cd22a5583659a3a0e7988a50e9b051416c46d1d8be31cc8'}
        check(set((proof.get('bound_vendor_sha256') or {}).values())==expected_vendors,'native vendor ELF hashes differ')
        check(summary['native_sdk']['sha256']==NATIVE_SDK_SHA,'native SDK hash differs')
        check(summary['helper']['sha256']==NATIVE_HELPER_SHA,'native helper hash differs')
    else:
        check(summary.get('graphs') is False,'oxide graphs must be false')
    return {'summary':str(out/'summary.json'),'summary_sha256':sha(out/'summary.json'),'stdout_sha256':sha(stdout),'stderr_sha256':sha(stderr),'process':record,'data':summary}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,default=MODEL)
    p.add_argument('--cases',type=Path,default=CASES)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pool',default=POOL)
    p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--iterations',type=int,default=200)
    p.add_argument('--aba-cycles',type=int,default=5)
    p.add_argument('--oxide-binary',type=Path)
    p.add_argument('--oxide-build',type=Path)
    p.add_argument('--kernel-root',type=Path,default=Path('/tmp/oxide-moe-root629-v8-public126-bundle-v1'))
    p.add_argument('--provider-root',type=Path,default=Path('/tmp/oxide-fa4-canonical26-derived-v4/bundle'))
    p.add_argument('--native-sdk',type=Path,default=NATIVE_SDK)
    p.add_argument('--native-helper',type=Path,default=NATIVE_HELPER)
    p.add_argument('--native-python',default='/opt/torch-sdk/bin/python3.12')
    p.add_argument('--cpu-only',action='store_true')
    p.add_argument('--skip-stable2k',action='store_true')
    p.add_argument('--justify-large',default='')
    args=p.parse_args()
    if args.iterations>200 and not args.justify_large:
        p.error('--iterations >200 requires --justify-large')
    args.output=args.output.resolve(); check(not args.output.exists(),f'refusing existing output {args.output}')
    args.output.mkdir(parents=True)
    report={'schema':'qwen3-moe-paired-perf/v1','windows_order':list(WINDOWS),'pool':args.pool,'environment':{'CUDA_VISIBLE_DEVICES':'0','one_engine_at_a_time':True,'gpu':'0','vendor_version':130200,'oxide_graphs':False,'native_graphs':True}}
    try:
        report['preflight']=preflight(args)
        if args.cpu_only:
            report['status']='cpu-prepared'; write(args.output/'report.json',report); print(json.dumps(report,indent=2)[:2000]); return 0
        cases_path=args.cases
        if args.skip_stable2k: cases_path=cases_without_stable(args.cases,args.output)
        windows=[]
        for name in WINDOWS:
            w=run_window(args,name,args.output/name,cases_path); windows.append(w)
            report['windows']=windows; write(args.output/'report.json',report)
        # Quality exact: T7/ragged129 vs existing Native gold; stable2k vs first native window.
        sys.path.insert(0,str(NATIVE_ROOT)); sys.path.insert(0,str(SOURCE_PACKET))
        import metrics
        native_report=read(NATIVE_REPORT); row=next(r for r in native_report['pool_reports'] if r['pool']==args.pool)
        gold={}
        for cid,nid in GOLD_CASES.items():
            c=next(x for x in row['checks'] if x['id']==nid)
            ref=read(c['native_report']); ref['layout']=metrics.nb.physical_layout(ref['physical_batch'],2048); gold[cid]=ref
        stable_ref=None
        for w in windows:
            if w['data']['backend']!='native' or stable_ref is not None: continue
            sc=next(c for c in w['data']['cases'] if c['id']=='stable2k')
            stable_ref=read(sc['report'])
        check(stable_ref is not None,'stable2k native reference missing')
        comparisons=[]
        for w in windows:
            for c in w['data']['cases']:
                cand=read(c['report'])
                ref=gold.get(c['id'],stable_ref)
                res=metrics.compare(ref,cand)
                comparisons.append({'window':w['window'],'backend':w['data']['backend'],'case':c['id'],'status':res['status'],'vectors':res['vectors'],'exact_vectors':res['exact_vectors'],'failed_vectors':res['failed_vectors']})
                check(res['status']=='passed' and res['failed_vectors']==0 and res['exact_vectors']==res['vectors'],f"quality failed {w['window']}/{c['id']}")
        report['comparisons']=comparisons
        # timing aggregation
        agg={}
        for cid in [c['id'] for c in read(cases_path)['cases']]:
            rows=[(w['window'],w['data']['backend'],next(c for c in w['data']['cases'] if c['id']==cid)) for w in windows]
            native=[r for r in rows if r[1]=='native']; oxide=[r for r in rows if r[1]=='oxide']
            def med(rows,key): return sorted(r[2][key] for r in rows)[len(rows)//2]
            agg[cid]={'windows':[{'window':r[0],'backend':r[1],'first_call_us':r[2]['first_call_us'],'p50_us':r[2]['p50_us'],'p95_us':r[2]['p95_us'],'mean_us':r[2]['mean_us'],'result_sha256':r[2]['result_sha256'],'report_sha256':r[2]['report_sha256']} for r in rows],'native_median_of_means_us':med(native,'mean_us'),'oxide_median_of_means_us':med(oxide,'mean_us'),'oxide_over_native_ratio':med(oxide,'mean_us')/med(native,'mean_us')}
        report['timing']=agg; report['status']='passed'
    except Exception as e:
        report['status']='failed'; report['error']=str(e)
    write(args.output/'report.json',report); print(json.dumps(report,indent=2)[:4000]); return 0 if report['status'] in ('cpu-prepared','passed') else 1

if __name__=='__main__': raise SystemExit(main())
