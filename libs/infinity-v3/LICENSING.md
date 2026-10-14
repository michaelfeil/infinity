# infinity_v3 licensing

Covered fork contributions are offered under your choice of:

- [PolyForm Noncommercial License 1.0.0](LICENSES/PolyForm-Noncommercial-1.0.0.txt), the unmodified standard license.
- [Infinity License 1.0 for Commercial Use](LICENSES/Infinity-License-1.0-for-Commercial-Use.txt), the custom commercial alternative.

The [Adoption Notice and complete license bundle](LICENSE) define the covered
rights, exclusions, and effective version. Adoption takes effect at the merge
that first introduces that notice into `mf/faster-tei`; that merge commit and
date identify the first covered source version and effective date. The
pre-adoption integration base is `ec11d07f4a0c354279366c616ac9184cc0c84850`.

**Infinity Core Contributors** describes each rights holder or authorized
licensor for their own contributions; it does not assign ownership or permit
one contributor to relicense another's work. **Michael Feil** authored the
commercial license text. PolyForm is published by the PolyForm Project. Both
alternatives are source-available licenses with use restrictions.

## Researchers: the standard PolyForm option

For noncommercial research, you can use the unmodified
[PolyForm Noncommercial 1.0.0 license](https://polyformproject.org/licenses/noncommercial/1.0.0).
Its SPDX identifier is
[`PolyForm-Noncommercial-1.0.0`](https://spdx.org/licenses/PolyForm-Noncommercial-1.0.0.html).
No separate permission request to Infinity is needed for covered rights when
your use complies with that license.

PolyForm permits noncommercial purposes on any platform. Its Personal Uses
clause covers specified personal activities without anticipated commercial
application. Its Noncommercial Organizations clause also expressly permits
use by charitable organizations, educational institutions, public research
organizations, public safety or health organizations, environmental
protection organizations, and government institutions, regardless of funding
sources or resulting funding obligations. Those permissions follow PolyForm's
text; they are not narrowed by the commercial alternative's definitions.

**No Infinity-specific conditions are added to PolyForm.** The commercial
platform blacklist, custom rewrite restriction, source-disclosure requirement,
and requirement to preserve both alternatives do not apply to uses relying
solely on PolyForm. PolyForm does not require publishing modifications or
licensing your modifications under both alternatives. When distributing,
follow PolyForm's Notices clause and any independently applicable third-party
terms. Institutional approval policies may still apply.

| Use | License treatment for covered rights |
| --- | --- |
| Noncommercial research on Modal, Together, Fireworks, or another listed platform | Allowed under PolyForm; there is no provider blacklist in that alternative. |
| Use by a university or public research organization, including funded work | Expressly permitted by PolyForm's Noncommercial Organizations clause. |
| A personal hobby or research project without anticipated commercial application | Covered by PolyForm's Personal Uses clause. |
| Commercial enterprise use not independently permitted by PolyForm, on owned hardware or an unlisted provider | Allowed under the commercial alternative, subject to its restrictions. |
| Commercial deployment not independently permitted by PolyForm, on a listed platform | Not authorized by the commercial alternative, including private endpoints and rented GPUs. |
| A for-profit provider sells an inference service to a university | The customer's PolyForm eligibility does not itself license the provider's separate commercial use. |
| Rewrite, including an agent-assisted C/Go port, for a listed platform's prohibited commercial deployment | Prohibited under the commercial alternative; PolyForm independently authorizes only its own permitted purposes. |
| Distribution relying solely on PolyForm | Follow PolyForm's notices; no added Infinity source-disclosure or dual-license condition. |

## Commercial Use Terms

The commercial alternative permits enterprise use on owned hardware or
unlisted providers, subject to its platform restrictions. Its Appendix A is
the fixed blacklist of 89 entries from the
[recorded OpenRouter directory snapshot](docs/licensing/openrouter-providers-2026-10-13.json),
retrieved on 2026-10-13 UTC. Four of the directory's 93 entries were excluded:
Azure, Baseten, Google Vertex, and Google AI Studio. Oracle Cloud is also
expressly permitted. Ordinary AWS infrastructure such as EC2 is permitted;
Amazon Bedrock is separately restricted. Future OpenRouter changes do not
alter the list, and no revenue test applies.

Under the commercial alternative, the named permitted providers may operate
their own deployments on hardware leased from listed operators if they
actually manage serving. This does not permit deployment inside a listed
service, including Bedrock. Routing, reselling, or relabeling cannot bypass
its restriction. Buying NVIDIA hardware, downloading models, storage, DNS,
transit, or unrelated API calls alone is not platform deployment.

The custom rewrite condition concerns licensed source, documentation, and
design obtained from them. General ideas, algorithms, independent development
without those materials, independently licensed rights, and nonwaivable legal
rights remain unaffected. This condition is not added to PolyForm.

## Existing rights and third-party material

The adoption adds the two alternatives only for rights whose holders
authorize them. Michael Feil adopts them for fork rights he owns or is
authorized to license. Another holder's existing INFOIL-only contributions
remain under INFOIL until that holder separately authorizes the alternatives.
Git authorship or maintainer access alone does not establish ownership or
relicensing authority.

Earlier Apache-2.0 and INFOIL grants remain available for the exact material
and rights they cover, including to later recipients. They do not license
new contributions first offered solely under this dual-license adoption.
INFOIL is not a third choice for those new contributions.

The historical INFOIL adoption notice and complete terms remain in the
[license bundle](LICENSE), alongside the retained
[Apache-2.0 text](LICENSES/Apache-2.0.txt). The original INFOIL
transition was effective on 2026-10-12, at `infinity_v3-infoil-1.0`, with
pre-adoption base `9486dd9ee89575442fda1a491c381046b37e116e`.

Inherited Hugging Face Text Embeddings Inference, previously Apache-covered
fork contributions, bundled components, and external dependencies retain
their independent grants and notices. Model weights and outputs do not become
covered by either alternative merely through use. See [NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and component notices.


## Contributions, redistribution, and packaging

New covered contributions intentionally submitted to this repository for
inclusion are offered under both the commercial license and PolyForm unless
different terms are expressly agreed before inclusion. This is an inbound
contribution policy, not an additional condition on using or redistributing
software under PolyForm. Contributors retain ownership.

Redistributions under the commercial alternative must retain both licenses,
Appendix A, the adoption notice, and third-party notices, identify changes,
and offer covered modifications under both alternatives. Include editable
source and build/install instructions, or a free download available for at
least three years after the last corresponding distribution. These are
conditions of the commercial alternative only. Neither alternative requires
publication merely because of private modifications or network use.

The root `LICENSE` bundles both complete alternatives and the unchanged
historical INFOIL/Apache bundle. First-party Cargo packages use this bundle
through `license-file`; independently licensed extension crates retain their
own declarations. Docker builds install the license bundle and component
notices at `/usr/share/licenses/infinity_v3/`; native LibTorch installations
retain their existing `share/licenses/tei_torch/` location.

The PolyForm file is a byte-for-byte copy of the
[official plain text](https://polyformproject.org/licenses/noncommercial/1.0.0.txt),
with SHA-256 `ffcca38841adb694b6f380647e15f17c446a4d1656fed51a1e2041d064c94cc8`.
The license text is not modified to add project-specific conditions.

This is not a complete audit of transitive dependencies, model weights,
CUDA libraries, or base images. Distributors must retain any additional
notices those artifacts require.

References: [SPDX's OR license-choice convention](https://spdx.github.io/spdx-spec/v2.3.1/SPDX-license-expressions/#d42-disjunctive-or-operator),
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0), and
[Copyright Office guidance on computer programs](https://www.copyright.gov/register/tx-programs.html).
