# Infinity V3 licensing

Infinity V3 uses the [Infinity Restricted MIT License 1.0](LICENSES/Infinity-Restricted-MIT-1.0.txt)
for covered original contributions. This is a custom, source-available license
based on MIT. Use its full name; it is not standard MIT or OSI open source.
Other Infinity packages retain their own licenses.

## Permissions

You may use, modify, sell, and redistribute the covered software, including in
proprietary products. You may keep modifications private and distribute binaries
without publishing source. Retain the complete license and required notices.
No revenue threshold, separate rewrite prohibition, or approval request applies
to permitted uses.

## Restricted providers

Use on the 25 providers in Appendix A of the [license](LICENSE) is prohibited
for every purpose, including research, education, evaluation, testing, and
personal use. This applies to operators and customers, including private
endpoints, rented GPUs, and listed-provider capacity used through intermediaries.
There are no research, institutional, or cloud-leasing exceptions under this
license. A convenient copy is in [RESTRICTED-PROVIDERS.txt](docs/licensing/RESTRICTED-PROVIDERS.txt).

| Deployment | Under this license |
| --- | --- |
| Your own hardware | Allowed |
| EC2, Azure, GCP, Baseten, or another unlisted provider | Allowed, unless using listed-provider capacity through an intermediary |
| A hosted Infinity service on permitted infrastructure | Allowed |
| A private endpoint or rented GPU on Modal | Restricted |
| Research, university, or personal use on a listed provider | Restricted |
| An unlisted intermediary using listed-provider compute | Restricted |

Nebius includes its hosting and compute services, including Token Factory.
Voyage AI identifies that service rather than unrelated MongoDB products.
The list is fixed for this license version.

## Scope and existing rights

Each licensor grants only rights they own or are authorized to license.
Inherited Text Embeddings Inference code, previously Apache-licensed
contributions, and third-party components retain their terms and notices.
Earlier MIT, Apache, INFOIL, PolyForm, and commercial grants remain available
for the exact material and rights they cover. The provider restriction does
not override independently granted permissions. Model weights, datasets, and
dependencies have separate terms.

See [LICENSES/README.md](LICENSES/README.md), [NOTICE](NOTICE), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Docker packages include the
license bundle at `/usr/share/licenses/infinity_v3/`; native LibTorch installs
include it at `share/licenses/tei_torch/`. The complete [LICENSE](LICENSE)
controls over this summary.
