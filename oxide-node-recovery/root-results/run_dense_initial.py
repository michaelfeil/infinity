import json,os,subprocess,sys,time
from pathlib import Path
P=Path("/tmp/oxide-dense-paired-e582-v1")
R=Path("/root/infinity-oxide-results/2026-11-02-requested-model-families")
O=Path("/tmp/oxide-dense-root-paired-initial-v1")
O.mkdir(exist_ok=False)
assert os.environ.get("CUDA_VISIBLE_DEVICES")=="0" and os.environ.get("INFINITY_OXIDE_ROOT_GPU0_LOCKED")=="1"
K="/tmp/infinity-oxide-parity-20261101/qwen35-public-f417-device99-v3-offline/kernels"
F="/tmp/oxide-fa4-canonical26-derived-v4/bundle"
rows=[("qwen3-reranker4b","qwen3-reranker-4b","cls")]+[("qwen3-4b","qwen3-4b",pool)for pool in ["cls","mean","last-token"]]
report={"status":"running","cases":[]}
def save():(O/"report.json").write_text(json.dumps(report,indent=2)+"\n")
save()
try:
 for case,native,pool in rows:
  label=case+"-"+pool;out=O/label;start=time.monotonic()
  argv=[sys.executable,str(P/"run_paired_root.py"),"--run","--case",case,"--api","factory","--pool",pool,"--kernel-root",K,"--fa4-root",F,"--output",str(out)]
  with (O/(label+".stdout")).open("x")as f:result=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(label+" actualRustcontrolfailed "+str(result.returncode))
  cargv=[sys.executable,str(P/"native_rust_compare.py"),"--native-case-dir",str(Path("/tmp/oxide-dense-native-root-quality-v2")/native),"--rust-report",str(out/"report.json"),"--pool",pool,"--output",str(O/(label+"-compare"))]
  with (O/(label+"-compare.stdout")).open("x")as f:comparison=subprocess.run(cargv,stdout=f,stderr=subprocess.STDOUT)
  if comparison.returncode:raise RuntimeError(label+" Nativecomparisonfailed "+str(comparison.returncode))
  report["cases"].append({"case":case,"pool":pool,"seconds":time.monotonic()-start,"output":str(out),"status":"passed"});save();print(json.dumps(report["cases"][-1]),flush=True)
 report["status"]="passed"
except BaseException as error:report.update(status="failed",error=str(error));raise
finally:save()
