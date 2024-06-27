.PHONY: generate tests

generate:
	pip install openapi-python-client==0.21.1
	 openapi-python-client generate  \
	  --url http://0.0.0.0:7997/openapi.json \
	  --config client_config.yaml \
	   --overwrite \
	   --custom-template-path=./template

tests:
	./run_tests_with_hook.sh
