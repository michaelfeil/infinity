# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
"""Use CuTe's supported AOT-only compilation control without a JIT engine.

FakeTensor prevents the FA4 forward call. It does not itself disable host JIT
entry resolution during compilation, which can load a foreign target image.
The DSL consumes ``no_jit_engine`` before binding kernel arguments and retains
the real compiled IR needed by the unchanged portable object exporter.
"""


def install_export_only_compile(cute):
    original = cute.compile
    state = {'supported_control': 'no_jit_engine=True', 'calls': []}

    def compile_export_only(*args, **kwargs):
        if kwargs.get('no_jit_engine') is False:
            raise RuntimeError('AOT export cannot request a runtime JIT engine')
        kwargs['no_jit_engine'] = True
        compiled = original(*args, **kwargs)
        if compiled.engine is not None or compiled.ir_module is None:
            raise RuntimeError('AOT compilation did not retain real IR without a JIT engine')
        state['calls'].append({'engine_is_none': compiled.engine is None,
                               'real_IR_present': compiled.ir_module is not None})
        return compiled

    cute.compile = compile_export_only
    return state
