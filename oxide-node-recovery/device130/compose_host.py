#!/usr/bin/env python3
"""Selective H5120 serving delta on public252789; never copy stale decoder."""
from pathlib import Path
import re
ROOT=Path('/tmp/infinity-oxide-qwen3-h5120-public252-v1-worktree')
OLD=Path('/tmp/infinity-oxide-qwen3-h5120-e582-v1-worktree')
REL=Path('libs/infinity-v3/backends/oxide/src')
def once(t,a,b):
    assert t.count(a)==1,(a,t.count(a));return t.replace(a,b,1)
def fnrange(t,name):
    m=re.search(r'(?m)^\s*(?:pub(?:\(crate\))?\s+)?(?:const\s+)?(?:unsafe\s+)?fn\s+'+name+r'\s*\(',t)
    assert m,name
    i=t.index('{',m.end());n=1;j=i+1
    while n:
        if t[j]=='{':n+=1
        elif t[j]=='}':n-=1
        j+=1
    return m.start(),j
def changefn(t,name,old,new):
    a,b=fnrange(t,name);return t[:a]+once(t[a:b],old,new)+t[b:]
def getfn(t,name):
    a,b=fnrange(t,name);return t[a:b]
p=ROOT/REL/'decoder.rs';t=p.read_text();donor=(OLD/REL/'decoder.rs').read_text()
t=once(t,'    Qwen3H4096Ordinary,\n','    Qwen3H4096Ordinary,\n    Qwen3H5120Ordinary,\n')
t=once(t,'                    4096 => Ok(Self::Qwen3H4096Ordinary),\n','                    4096 => Ok(Self::Qwen3H4096Ordinary),\n                    5120 => qwen3_h5120_admission(config).map(|()| Self::Qwen3H5120Ordinary),\n')
t=once(t,'only Qwen3 hidden_size 1024, 2048, 2560 or 4096;','only Qwen3 hidden_size 1024, 2048, 2560, 4096 or 5120;')
t=changefn(t,'rms_export','            Self::Qwen3H2048Ordinary | Self::Qwen3H2560Ordinary => "infinity_rms_norm",','            Self::Qwen3H2048Ordinary | Self::Qwen3H2560Ordinary => "infinity_rms_norm",\n            Self::Qwen3H5120Ordinary => "infinity_rms_norm_5120_aten",')
t=changefn(t,'add_rms_export','Self::Qwen3H2048Ordinary | Self::Qwen3H2560Ordinary =>','Self::Qwen3H2048Ordinary | Self::Qwen3H2560Ordinary | Self::Qwen3H5120Ordinary =>')
for name in ['rope_export','qk_norm_rope_export','activation_export','norm_geometry']:
    t=changefn(t,name,'            | Self::Qwen3H4096Ordinary\n','            | Self::Qwen3H4096Ordinary\n            | Self::Qwen3H5120Ordinary\n')
pos=t.index('/// Dense Qwen3 fused per-head')
t=t[:pos]+'''/// Ordinary H5120 RMS uses the qualified ATen128 vectorized reduction.
/// Add-RMS remains on its separately qualified generic256 recipe.
const H5120_RMS_GEOMETRY: NormGeometry = NormGeometry {
    rows_per_cta: 1,
    threads: 128,
};
'''+t[pos:]
a,b=fnrange(t,'norm_geometry')
t=t[:b]+'''

    const fn rms_geometry(self) -> NormGeometry {
        match self {
            Self::Qwen3H5120Ordinary => H5120_RMS_GEOMETRY,
            _ => self.norm_geometry(),
        }
    }
'''+t[b:]
pos=t.index('/// Pooled-mean admission')
helper=getfn(donor,'qwen3_h5120_admission').lstrip()
t=t[:pos]+'''/// H5120 Qwen3 dense uses the two Native d128/GQA topologies. Query
/// width is independently checked as heads*head_dim; Q64 is wider than H5120.
'''+helper+'\n\n'+t[pos:]
t=once(t,'    norm_geometry: NormGeometry,','    rms_geometry: NormGeometry,\n    norm_geometry: NormGeometry,')
t=once(t,'            norm_geometry: math.norm_geometry(),','            rms_geometry: math.rms_geometry(),\n            norm_geometry: math.norm_geometry(),')
t=once(t,'(rms_rows.div_ceil(k.norm_geometry.rows_per_cta), 1, 1),\n            k.norm_geometry.threads,','(rms_rows.div_ceil(k.rms_geometry.rows_per_cta), 1, 1),\n            k.rms_geometry.threads,')
# Reuse source-admission/checked-arena tests; their old RMS expectation is changed
# to the newly proven exact recipe. These are not trained-checkpoint claims.
tests=[]
for name in ['qwen3_h5120_ordinary_recipe_keeps_native_generic_math_and_exact_qkv_grouping','qwen3_h5120_admission_is_limited_to_the_two_native_topologies','qwen3_h5120_workspace_bytes_follow_query_width_and_positive_intermediate']:
    f=getfn(donor,name)
    if name.startswith('qwen3_h5120_ordinary'):
        f=f.replace('"infinity_rms_norm"','"infinity_rms_norm_5120_aten"')
        f=f.replace('qwen3_h5120_ordinary_recipe_keeps_native_generic_math_and_exact_qkv_grouping','qwen3_h5120_qualified_aten_rms_keeps_add256_and_checked_qkv_grouping')
        marker='''            assert_eq!(c.q_width(), heads * 128);'''
        f=once(f,marker,'''            assert_eq!(DecoderMath::Qwen3H5120Ordinary.rms_geometry(), H5120_RMS_GEOMETRY);
            assert_eq!(DecoderMath::Qwen3H5120Ordinary.rms_geometry().threads, 128);
            assert_eq!(DecoderMath::Qwen3H5120Ordinary.norm_geometry().threads, 256);
'''+marker)
    tests.append('    #[test]\n'+f+'\n')
pos=t.index('    #[test]\n    fn qwen3_4b_actual_config_selects_h2560_ordinary_geometry')
t=t[:pos]+'\n'.join(tests)+'\n'+t[pos:]
t=t.replace('serde_json::json!({"hidden_size": 5120}),\n                "hidden_size 1024, 2048, 2560 or 4096",','serde_json::json!({"hidden_size": 6144}),\n                "hidden_size 1024, 2048, 2560, 4096 or 5120",')
p.write_text(t)
p=ROOT/REL/'decoder_model.rs';t=p.read_text();d=(OLD/REL/'decoder_model.rs').read_text()
f=getfn(d,'qwen3_h5120_positive_intermediate_and_independent_query_widths')
pos=t.index('    #[test]\n    fn ',t.index('fn qwen3_native_dense_hidden_widths')) if 'fn qwen3_native_dense_hidden_widths' in t else t.rfind('\n}')
t=t[:pos]+'    #[test]\n'+f+'\n'+t[pos:]
p.write_text(t)
print(ROOT)
