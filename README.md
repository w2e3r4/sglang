# SGLang PR #37785 validation evidence

Results for the public v0.5.23 release baseline plus the extend/decode patch in `tested-release-fix.diff`.

- [Accuracy results and limitations](accuracy-results.md)
- [Full public reproduction bundle](fa3-descale-public-reviewed.tar.gz)
- [Metrics](metrics.json)
- [Tested release patch](tested-release-fix.diff)
- [Follow-up applied to the PR branch](pr-followup.patch)

The PR source update is commit `67a44c8639026fb35ebb79eb7bc0e9994263d7e0`. Its production decode change matches the tested release port; the PR commit itself was not run on GPU. This evidence branch is separate from the PR source branch.

The archive contains launch/evaluation commands, original evaluator output, normalized per-sample predictions, metrics and supporting logs. Verify the archive against `SHA256SUMS`.
