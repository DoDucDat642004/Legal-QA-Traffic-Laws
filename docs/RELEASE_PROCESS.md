# Release And Deployment Process

This project uses two publication targets:

- GitHub for source code, review history, CI, issues, and lightweight evaluation sets.
- Hugging Face Spaces for the runnable demo and runtime artifacts needed by the Space.

Do not push directly from a dirty working tree. Do not commit real secrets, local caches, or generated experiment reports.

## Branching

Use feature branches for development:

```bash
git switch -c feature/<short-topic>
```

Before opening a pull request, sync with the remote branch:

```bash
git fetch origin
git rebase origin/feature/data-pipeline
```

If the branch contains duplicated cherry-picked commits, prefer creating a clean release branch from the current remote and cherry-picking only the intended commits.

## Commit Boundaries

Keep commits reviewable:

- Evaluation logic changes.
- Retrieval behavior changes.
- Dataset or boost regression fixtures.
- Deployment/configuration changes.
- Documentation-only changes.

Avoid combining generated data refreshes with application logic changes.

## Pre-Push Gate

Run:

```bash
scripts/prepush_check.sh
```

Enable the tracked local hook once per clone:

```bash
git config core.hooksPath .githooks
```

The script checks:

- Real secret patterns.
- Python and shell syntax.
- Git LFS tracking status.
- Local boost retrieval regression when the required data exists.

## GitHub Release Flow

1. Confirm the branch is clean except for intended changes:

   ```bash
   git status --short --branch
   ```

2. Stage intentionally:

   ```bash
   git add <files>
   git diff --cached --stat
   git diff --cached
   ```

3. Commit with an action-oriented message:

   ```bash
   git commit -m "Improve boost retrieval regression"
   ```

4. Push the feature branch:

   ```bash
   git push origin feature/<short-topic>
   ```

5. Open a pull request with:

   - Summary of behavior changes.
   - Validation output.
   - Data/model migration notes.
   - Known risks.

## Hugging Face Space Flow

Secrets belong in Space settings, not in Git:

- `GEMINI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY` when needed

Before pushing to Hugging Face, confirm the remote URL does not contain a token:

```bash
git remote -v
```

Expected format:

```text
hf https://huggingface.co/spaces/<owner>/<space> (fetch)
hf https://huggingface.co/spaces/<owner>/<space> (push)
```

Push only after the GitHub branch has passed the pre-push gate:

```bash
git push hf <branch>:main
```

## Data And Artifact Policy

Use Git LFS for required runtime assets that must be versioned with the app. Keep generated reports, caches, local vector stores, and temporary outputs out of Git.

Recommended split:

- Source code and small regression fixtures: GitHub.
- Space runtime files and required LFS assets: Hugging Face Space.
- Large reusable models or datasets shared across apps: separate Hugging Face Model/Dataset repositories.

## Incident Response

If a token appears in a remote URL, command output, commit, or log:

1. Revoke or rotate the token in the provider dashboard.
2. Remove the token from local config:

   ```bash
   git remote set-url hf https://huggingface.co/spaces/<owner>/<space>
   ```

3. Search the repository:

   ```bash
   rg "hf_[A-Za-z0-9]+|sk-[A-Za-z0-9_-]+|AIza[0-9A-Za-z_-]+" .
   ```

4. If the token was committed, rewrite history before publishing and force-push only with explicit team agreement.
