import json,os,subprocess,sys,time
from pathlib import Path
P=Path("/tmp/oxide-all-model-feature-matrix-v1/qwen3-moe-serving-v1")
R=Path("/root/infinity-oxide-results/2026-11-02-requested-model-families")
assert os.environ.get("CUDA_VISIBLE_DEVICES")=="0" and os.environ.get("INFINITY_OXIDE_ROOT_GPU0_LOCKED")=="1"
for mode in ["plain","memcheck"]:
 out=P/("root-"+mode+"-v3")
 assert not out.exists(),str(out)
 argv=[sys.executable,str(P/"root/run_unified_v3.py"),"--mode",mode,"--cpu-run","184-191","--output",str(out)]
 start=time.monotonic()
 with (R/("moe-"+mode+"-v3.stdout")).open("x") as f:
  result=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT)
 print(json.dumps({"mode":mode,"code":result.returncode,"seconds":time.monotonic()-start,"report":str(out/"report.json")}),flush=True)
 if result.returncode:sys.exit(result.returncode)
