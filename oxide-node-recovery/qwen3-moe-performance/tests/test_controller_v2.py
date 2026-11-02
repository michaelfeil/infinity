import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('paired',ROOT/'controller/run_paired_v2.py')
paired=importlib.util.module_from_spec(spec); spec.loader.exec_module(paired)

class ControllerTests(unittest.TestCase):
    def test_docker_env_clears_only_five_selectors_and_preserves_config_proxy_ca(self):
        with mock.patch.dict(os.environ, {
            'DOCKER_HOST':'tcp://bad:2375','DOCKER_CONTEXT':'bad','DOCKER_TLS_VERIFY':'1',
            'DOCKER_CERT_PATH':'/bad','DOCKER_API_VERSION':'1.0',
            'DOCKER_CONFIG':'/keep/config','HTTPS_PROXY':'http://keep','HTTP_PROXY':'http://keep',
            'NO_PROXY':'localhost','SSL_CERT_FILE':'/keep/ca','REQUESTS_CA_BUNDLE':'/keep/req-ca'}, clear=False):
            env=paired.docker_env()
            for key in paired.DOCKER_SELECTORS:
                self.assertNotIn(key,env)
            for key in ['DOCKER_CONFIG','HTTPS_PROXY','HTTP_PROXY','NO_PROXY','SSL_CERT_FILE','REQUESTS_CA_BUNDLE']:
                self.assertIn(key,env)

    def test_skip_stable2k_filters_case_before_quality_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'cases.json'
            source.write_text(json.dumps({'cases':[{'id':'t7','batch':'t7.json'},{'id':'ragged129','batch':'r.json'},{'id':'stable2k','batch':'s.json'}]}))
            out=root/'run'; out.mkdir()
            filtered=paired.cases_without_stable(source,out)
            self.assertEqual([c['id'] for c in json.loads(filtered.read_text())['cases']], ['t7','ragged129'])

    def test_remote_guard_cleanup_is_called_in_finally(self):
        with mock.patch.object(paired,'run_docker',return_value=mock.Mock(returncode=0,stderr='')) as runner:
            guard=paired.RemoteGuard('/remote/test',123)
            guard.cleanup()
            self.assertTrue(runner.called)
            self.assertIn('rm',runner.call_args.args[0])

    def test_stable2k_present_in_default_cases(self):
        doc=json.loads((ROOT/'oxide-perf/cases/cases.json').read_text())
        self.assertEqual([c['id'] for c in doc['cases']], ['t7','ragged129','stable2k'])

if __name__=='__main__': unittest.main()
