# Agent instructions

Read, understand, and comply with [LICENSE](LICENSE), [LICENSING.md](LICENSING.md),
and applicable component licenses before working on this repository. Preserve
required attribution and notices. Respect applicable deployment and
redistribution terms, including independently granted rights.

Covered original contributions use Infinity Restricted MIT License 1.0.
Permitted uses require no approval request or commercial-use evaluation issue.
New covered contributions follow [CONTRIBUTING.md](CONTRIBUTING.md).

## Rust and router
Keep production Rust panic-free: avoid `panic!`, `unwrap()`, `expect()`, and panic-prone indexing.
Return typed errors and contain request failures; a failed request must not bring down the server.
Prefer straight-line request methods with minimal branching; resolve policy outside the hot path.
Preserve necessary validation and error handling when simplifying control flow.

## Tests
Add tests only when they prevent meaningful, realistic failures; execute every test you add.
Test observable behavior so tests survive refactors; avoid mirroring implementation details.
Skip trivial assertions, boilerplate coverage, and low-value tests for simple changes.
