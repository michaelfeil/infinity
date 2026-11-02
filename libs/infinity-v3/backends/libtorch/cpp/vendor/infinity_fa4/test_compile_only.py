# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
"""CPU checks for the AOT control that prevents foreign-target JIT loading."""
from types import SimpleNamespace
import unittest
from compile_only import install_export_only_compile

class CompileOnlyTests(unittest.TestCase):
    def test_real_ir_is_retained_and_jit_engine_disabled(self):
        calls=[]
        def compile(*args,**kwargs):
            calls.append((args,kwargs));return SimpleNamespace(engine=None,ir_module=object())
        cute=SimpleNamespace(compile=compile);state=install_export_only_compile(cute)
        compiled=cute.compile('kernel',target='sm_100a')
        self.assertIsNone(compiled.engine)
        self.assertEqual(calls,[(('kernel',),{'target':'sm_100a','no_jit_engine':True})])
        self.assertEqual(state['calls'],[{'engine_is_none':True,'real_IR_present':True}])
    def test_explicit_runtime_jit_request_is_rejected(self):
        def forbidden(*args,**kwargs):raise AssertionError('runtime JIT request reached compiler')
        cute=SimpleNamespace(compile=forbidden);install_export_only_compile(cute)
        with self.assertRaises(RuntimeError):cute.compile('kernel',no_jit_engine=False)
    def test_missing_ir_or_runtime_engine_rejected_before_export(self):
        for result in [SimpleNamespace(engine=object(),ir_module=object()),SimpleNamespace(engine=None,ir_module=None)]:
            with self.subTest(result=result):
                cute=SimpleNamespace(compile=lambda *a,**k:result);state=install_export_only_compile(cute)
                with self.assertRaises(RuntimeError):cute.compile('kernel')
                self.assertEqual(state['calls'],[])
if __name__=='__main__':unittest.main()
