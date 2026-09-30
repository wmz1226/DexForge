# Third-party software

The simulator in `third_party/comfree_warp` includes ComFree contact simulation,
Gaussian collision queries, native adjoints, and MuJoCo Warp runtime code.
Its [licensing notice](third_party/comfree_warp/LICENSE) distinguishes:

- [ComFree Core Academic Research License](third_party/comfree_warp/comfree_warp/comfree_core/LICENSE)
  for `comfree_core/`, except where individual files state otherwise.
- [Apache License 2.0](third_party/comfree_warp/LICENSES/Apache-2.0.txt)
  for MuJoCo Warp and identified derived files.

Local adaptations include Gaussian collision, native adjoints, and fixed-order
state-gradient accumulation. Upstream copyright and license notices are retained.
The project's MIT license does not replace these dependency licenses.

Python dependencies in the root [requirements.txt](../requirements.txt) are
installed separately and retain their own licenses. See the root
[third-party notices](../THIRD_PARTY_NOTICES.md) for data and asset attribution.
