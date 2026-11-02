integration-tests:
	cargo test

cuda-integration-tests:
	cargo test --features libtorch --profile release-debug

integration-tests-review:
	cargo insta test --review

cuda-integration-tests-review:
	cargo insta test --review --features libtorch --profile release-debug

.PHONY: oxide-tests
oxide-tests:
	python3 -m unittest discover -s backends/oxide/device/tests
	cargo test --locked -p infinity-backend-oxide --all-targets
