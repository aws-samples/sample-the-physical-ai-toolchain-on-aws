# GR00T merge into `development` — report

Nothing pushed. Local `development` is now 14069d7. Waiting for human approval to push.

## Outcome

- Both merges completed with no conflicts. `git merge --no-ff` (ort strategy) auto-merged both. The only file both branches touched was `containers/gr00t-training/Dockerfile.n17`, and their hunks don't overlap.
  - 8ffa223 `Merge gr00t-batch-training into development`
  - 14069d7 `Merge gr00t-image-build into development`
- Conflicts and resolutions: none. The suspected conflict sites came out like this:
  - `Dockerfile.n17`: auto-merged. image-build changed lines 14–21 (FROM). batch-training changed lines 69+ (COPY/ENV/entrypoint).
  - `buildspec-n17.yml`: only batch-training touched it, so it is batch-training's version byte for byte.
  - `isaac-lab-on-aws/infra/batch.tf`: only image-build changed it, on top of 67d8f2d. The merged file has `min_vcpus = 0` and no `desired_vcpus` line. Byte-identical to gr00t-image-build.
  - `isaac-gr00t-on-aws/`: the branches touched different files (batch-training: `batch-training-guide.md`, `infra/batch.tf`, `infra/outputs.tf`; image-build: `event-build-image.sh`). No overlap.
- The merged `Dockerfile.n17` equals gr00t-batch-training's Dockerfile plus only the `ARG BASE_IMAGE` parameterisation. `git diff gr00t-batch-training development -- containers/gr00t-training/Dockerfile.n17` gives:

```diff
@@ -14,7 +14,11 @@
 #
 # Requires NGC API key to pull the base image (nvcr.io/nvidia/pytorch).
-FROM nvcr.io/nvidia/pytorch:25.04-py3
+# Base layers come from NGC by default. The workshop CloudFormation template can
+# override this with a pre-mirrored image in a central ECR, so participant accounts
+# never need an NGC key. See isaac-gr00t-on-aws/event-build-image.sh.
+ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3
+FROM ${BASE_IMAGE}
```

  I diffed both versions before merging (`git diff development...gr00t-image-build` and `git diff origin/development...gr00t-batch-training`). gr00t-image-build changed nothing else in the Dockerfile: only these 3 comment lines and the ARG/FROM.
- The `BASE_IMAGE` default is the verified build's base. batch-training's `buildspec-n17.yml` runs `docker build -f containers/gr00t-training/Dockerfile.n17 -t groot-training-n17:n17 .` with no `--build-arg`. So the verified n17-20261006 image (sha256:2638a3eb…8de1) was built `FROM nvcr.io/nvidia/pytorch:25.04-py3`, which is exactly the ARG default. `event-build-image.sh` on its NGC path also passes `BASE_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3`.
- Every other file batch-training owns is identical to the branch tip in the merged tree. `git diff --stat gr00t-batch-training development` lists only `Dockerfile.n17` (+ARG), `event-build-image.sh` (new) and `isaac-lab-on-aws/infra/batch.tf` (67d8f2d + 856a386). So `train_entrypoint_n17.py` and `batch-train-entrypoint-n17.sh`, which were verified byte-identical to the baked `/opt/ml/code/*` in the verified image, are unchanged.

### Caveats on "same image" (no action taken)

1. `nvcr.io/nvidia/pytorch:25.04-py3` is a tag, not a digest. If NVIDIA re-publishes it, a per-account rebuild could get different base layers. The verified build has the same exposure.
2. On the `CENTRAL_BASE_IMAGE_URI` path, the base is whatever was mirrored. It only matches the verified build if the mirror was copied from `nvcr.io/nvidia/pytorch:25.04-py3`.
3. The Dockerfile also clones Isaac-GR00T at the pinned `GROOT_COMMIT` and runs `uv sync`. A rebuild is reproducible in source but not guaranteed bit-identical. Expect a different image digest per account.
4. The repo root has no `.dockerignore`, so the whole repo is sent as build context. The verified build did the same, so this isn't a regression.

## Verification (merged tree at 14069d7)

| Check | Result |
|---|---|
| Pre-merge `git status` | Clean. There was no `.DS_Store` either. |
| `bash -n` on event-build-image.sh, batch-train-entrypoint-n17.sh, cosmos-on-aws/event-provision.sh | All OK |
| `shellcheck -S warning` on the same 3 files | All exit 0, no findings. shellcheck isn't installed locally, so it ran via ephemeral `uvx --from shellcheck-py==0.10.0.1` (ShellCheck 0.10.0). Nothing was installed into the repo. |
| `python3 -m py_compile` (Python 3.14.7) on all 5 touched .py files | All OK: `containers/gr00t-training/g1_dex3_modality_config.py`, `containers/gr00t-training/train_entrypoint_n17.py`, `training/gr00t/prepare_g1_dex3_dataset.py`, `training/gr00t/submit_batch_training_n17.py`, `training/gr00t/validate_lerobot_gr00t_dataset.py`. The .pyc output went to /tmp and was deleted. |
| Dockerfile COPY sources exist repo-root-relative and are git-tracked | All 4 exist: `containers/gr00t-training/train_entrypoint_n17.py` (L72), `containers/gr00t-training/batch-train-entrypoint-n17.sh` (L73), `containers/gr00t-training/ur3_modality_config.py` (L82), `containers/gr00t-training/g1_dex3_modality_config.py` (L83). L121 `COPY <<'EOF' /opt/ml/code/entrypoint.sh` is an inline heredoc with no source. There are no ADD lines. |
| event-build-image.sh Dockerfile path | Resolves. It sets `REPO_ROOT=$SCRIPT_DIR/..`, runs `cd "$REPO_ROOT"`, then `docker build -f containers/gr00t-training/Dockerfile.n17 ... "$REPO_ROOT"`. `<repo>/containers/gr00t-training/Dockerfile.n17` exists. The script is mode 100755. |
| Env-var contract | Consistent. No misspelled variants found (see table below). |
| Basename dispatch in inline entrypoint | Present: `if [ "$(basename -- "${1:-}")" = "batch-train" ]; then` |
| `ENV TUNE_VISUAL=0` / `ENV EMBODIMENT_CONFIG=ur3` | Both present in Dockerfile.n17 |
| `terraform fmt -check` (v1.16.4) | Exit 0 in isaac-gr00t-on-aws/infra, isaac-lab-on-aws/infra and cosmos-on-aws/infra |
| `terraform init -backend=false` | Exit 0 in all three |
| `terraform validate` | "Success! The configuration is valid." in all three |
| plan / apply | Not run, as instructed |
| Secret / account-id scan of `origin/development..development` changed lines | `hf_[A-Za-z0-9]{20,}`: 0. `AKIA`: 0. `804152302157`: 0. `nvapi-`: 1 hit, justified (see below). |

### Env-var occurrence counts

| File | EMBODIMENT_CONFIG | MODALITY_CONFIG_PATH | HF_TOKEN_SECRET_ID | TUNE_VISUAL |
|---|---|---|---|---|
| batch-train-entrypoint-n17.sh | 5 | 5 | 2 | 4 |
| train_entrypoint_n17.py | 12 | 6 | 0 | 9 |
| Dockerfile.n17 | 2 | 1 | 1 | 2 |
| submit_batch_training_n17.py | 1 | 0 | 1 | 5 |
| isaac-gr00t-on-aws/infra/batch.tf | 1 | 0 | 3 | 4 |
| isaac-gr00t-on-aws/batch-training-guide.md | 14 | 1 | 7 | 19 |

The zeros are by design:

- `HF_TOKEN_SECRET_ID` is resolved in the shell entrypoint. The shell then exports `HF_TOKEN` to the Python trainer.
- `MODALITY_CONFIG_PATH` is an optional override, so the submit script and Terraform don't set it.

Across these 6 files, the only UPPER_SNAKE tokens that look like these names are the four correct spellings plus unrelated names: `DEFAULT_TUNE_VISUAL`, `EMBODIMENT_CONFIG_MODULES`, `EMBODIMENT_TAG`, `NEW_EMBODIMENT`, `HF_TOKEN`, `YOUR_HF_TOKEN`. How each file consumes them:

- Shell: `${EMBODIMENT_CONFIG:-ur3}`, `${MODALITY_CONFIG_PATH:-}`, `${TUNE_VISUAL:-0}`, `${HF_TOKEN_SECRET_ID:-}`
- Python: `os.environ.get("MODALITY_CONFIG_PATH")`, `os.environ.get("EMBODIMENT_CONFIG","ur3")`, `_env_flag("TUNE_VISUAL", False)`
- Submit script: sets `EMBODIMENT_CONFIG`, `TUNE_VISUAL`, `HF_TOKEN_SECRET_ID`
- Terraform job definition: sets `TUNE_VISUAL=0`, `EMBODIMENT_CONFIG=g1_dex3`, `HF_TOKEN_SECRET_ID=${var.project_name}/hf-token`

### Scan hits

- `nvapi-`: `isaac-gr00t-on-aws/batch-training-guide.md:370` reads "A grep for `nvapi-` across the full build log of the validated build returns 0 matches." This is prose describing a check. It contains no key, so I kept it.
- `804152302157` doesn't appear in this diff. It already exists on `origin/development`, outside this change: `cosmos-on-aws/ec2-deployment-guide.md:368` and `isaac-sim-on-aws/headless-mode-guide.md:41,73,92`. This push doesn't publish them, but they are already public on origin. Flagging them for a separate cleanup.

## What `git push origin development` would publish

`git log --oneline origin/development..development` (17 commits: 2 merges, 1 already-local dev fix, 12 batch-training, 2 image-build):

```
14069d7 Merge gr00t-image-build into development
8ffa223 Merge gr00t-batch-training into development
2416068 docs: correct the stale single-GPU compute-environment prerequisite
683a90e docs: record the measured single-GPU smoke test in the Batch training guide
f289bbf fix(batch): make the Terraform GPU defaults self-consistent and cost commands path-aware
856a386 fix(isaac-lab): stop Terraform fighting Batch over desired_vcpus
edd08f8 feat: add per-account GR00T N1.7 image build bootstrap
6ae22e9 fix: refuse a global batch the job definition's GPU count cannot divide
586fee7 docs: make single-GPU the primary Batch path and document the required job-definition fields
fdd1c85 feat: make vision-tower tuning configurable and default the job shape to one GPU
67d8f2d fix(isaac-lab): scale the RL compute environment from zero
a42eeda fix: dispatch the Batch entrypoint on basename so the job definition command reaches it
7aba556 docs: quote the measured 22.6 GB image size in the Batch scripts
42f0e77 chore: source HF token from Secrets Manager and rewrite the Batch training guide for N1.7 + G1 Edu
b71f878 feat: AWS Batch submit-and-watch script for GR00T N1.7 training
1a577d7 fix: make the N1.7 container embodiment-selectable instead of UR3-only
700fac6 feat: G1 Edu Dex3 data prep and GR00T loader validation for N1.7 training
```

`git diff --stat origin/development..development`:

```
 containers/gr00t-training/Dockerfile.n17           |   36 +-
 .../gr00t-training/batch-train-entrypoint-n17.sh   |   49 +-
 containers/gr00t-training/buildspec-n17.yml        |   43 +-
 .../gr00t-training/g1_dex3_modality_config.py      |  139 +++
 containers/gr00t-training/train_entrypoint_n17.py  |  206 +++-
 isaac-gr00t-on-aws/batch-training-guide.md         | 1046 ++++++++++++++++----
 isaac-gr00t-on-aws/event-build-image.sh            |  122 +++
 isaac-gr00t-on-aws/infra/batch.tf                  |  180 +++-
 isaac-gr00t-on-aws/infra/outputs.tf                |    5 +
 isaac-lab-on-aws/infra/batch.tf                    |   30 +-
 training/gr00t/prepare_g1_dex3_dataset.py          |  440 ++++++++
 training/gr00t/submit_batch_training_n17.py        |  559 +++++++++++
 training/gr00t/validate_lerobot_gr00t_dataset.py   |  567 +++++++++++
 13 files changed, 3183 insertions(+), 239 deletions(-)
```

## Backup tags (local only, not pushed)

| Tag | Commit |
|---|---|
| `backup/development-before-gr00t-merge` | 67d8f2d |
| `backup/gr00t-image-build-tip` | 856a386 |
| `backup/gr00t-batch-training-tip` | 2416068 |

Undo, if needed (not run): `git -C /Users/asriaws/Codebase/sample-aws-physical-ai-toolchain reset --hard backup/development-before-gr00t-merge` while on `development`.

## State left behind

- Main checkout is on `development` at 14069d7. Working tree is clean apart from this untracked report under `.agents/`, which isn't gitignored. `git push` sends commits only, so the report won't be published.
- Both worktrees are untouched and clean: `gr00t-batch-training` at 2416068 and `gr00t-image-build` at 856a386. Both branches are kept.
- `terraform init` created a gitignored `.terraform/` and `.terraform.lock.hcl` in `isaac-lab-on-aws/infra`, which didn't exist before. I removed both. The ones in `isaac-gr00t-on-aws/infra` and `cosmos-on-aws/infra` already existed and are gitignored, so I left them.
- No AWS calls, CodeBuild builds, Batch jobs, or terraform plan/apply. The workshop content repo wasn't touched.

## Push command for the human to approve

```bash
git -C /Users/asriaws/Codebase/sample-aws-physical-ai-toolchain push origin development
```
