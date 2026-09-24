# Public Repository Path Sanitization

This repository is a public research snapshot of the model-training workflow.

Machine-specific filesystem paths from the original training server were
sanitized before publication.

Generic examples:

- Original project paths were replaced with:
  `/path/to/fire_detact_training/...`

- Original dataset paths were replaced with:
  `/path/to/fire_datasets/...`

Python training scripts that previously assumed the original local project
directory were adjusted to resolve the repository root from the script
location where applicable.

These changes affect filesystem references only.

They do **not** alter:

- training metrics;
- evaluation metrics;
- model architecture;
- model-selection results;
- SHA256 model hashes;
- final-test results.

The frozen model artifacts remain identified by the SHA256 values recorded
in `model/SHA256SUMS.txt`.
