# Agent instructions

Read, understand, and comply with [LICENSE](LICENSE), [LICENSING.md](LICENSING.md),
and applicable component licenses before working on this repository. Preserve
required attribution and notices. Respect applicable deployment and
redistribution terms, including independently granted rights.

Covered original contributions use Infinity Restricted MIT License 1.0.
Permitted uses require no approval request or commercial-use evaluation issue.
New covered contributions follow [CONTRIBUTING.md](CONTRIBUTING.md).

## Kernel selection
Prefer common, proven kernels such as FA4 across supported GPU families and providers.
Use a shared dispatch policy; avoid architecture-specific tuning and complicated fallback chains.
Add special paths only when required for correctness or an unsupported capability.
