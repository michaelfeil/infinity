integration-tests:
	cargo test

cuda-integration-tests:
	cargo test --features libtorch --profile release-debug

integration-tests-review:
	cargo insta test --review

cuda-integration-tests-review:
	cargo insta test --review --features libtorch --profile release-debug
