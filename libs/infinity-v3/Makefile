integration-tests:
	cargo test

cuda-integration-tests:
	cargo test -F infinity_v3-backend-candle/cuda -F infinity_v3-backend-candle/flash-attn -F infinity_v3-router/candle-cuda --profile release-debug

integration-tests-review:
	cargo insta test --review

cuda-integration-tests-review:
	cargo insta test --review --features "infinity_v3-backend-candle/cuda infinity_v3-backend-candle/flash-attn infinity_v3-router/candle-cuda" --profile release-debug
