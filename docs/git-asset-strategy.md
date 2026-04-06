# Git Asset Strategy

This note documents how to keep this GitHub Pages repository from growing too
quickly when PDFs, images, videos, and test artifacts are involved.

## 1. Core rule

This repository is a GitHub Pages site, so files that the website must serve
directly should stay in the normal Git repository.

GitHub's documentation states that Git LFS cannot be used with GitHub Pages
sites:

<https://docs.github.com/zh/enterprise-server%403.20/repositories/working-with-files/managing-large-files/about-git-large-file-storage>

That means site-facing assets such as `assets/pdf/*.pdf`, `assets/img/*`, and
`assets/video/*` should not be moved to LFS as the default solution.

## 2. What to do instead

Use this decision rule before adding or updating a large file:

1. If the file must be rendered or downloaded from this site, keep it in the
   repository.
2. If the file is a local build artifact, screenshot, test output, or cache,
   ignore it with `.gitignore`.
3. If the file is a large downloadable attachment and a stable external host
   already exists, prefer linking out instead of storing the binary here.
4. If a file changes frequently and is only an attachment, consider moving it
   to an external location such as a release asset or object storage.

## 3. Current findings in this repository

### Safe to keep locally

These are current site assets that appear to be part of the published site:

- `assets/pdf/recurrent_transformer.pdf`
- `assets/pdf/LEDiT_poster.pdf`
- `assets/pdf/JSEN3103042.pdf`
- `assets/pdf/resume.pdf`
- `assets/img/profile.png`
- `assets/img/world-model/genie3.gif`

Notes:

- The publication entries in `_bibliography/papers.bib` point to local files for
  `recurrent_transformer.pdf`, `LEDiT_poster.pdf`, and `JSEN3103042.pdf`.
- `_pages/cv.md` points to `resume.pdf`.

### Good candidates for external hosting

These are large binaries that are not essential as repository-managed source
files:

- `assets/video/tutorial_al_folio.mp4`
- `assets/video/pexels-engin-akyurt-6069112-960x540-30fps.mp4`

Reasoning:

- Tutorial and stock/demo videos are bulky, and they are not core source files.
- If they are still needed publicly, serving them from GitHub Releases or a
  dedicated file host is usually cleaner than keeping repeated binary history in
  the main site repository.

### Strong cleanup candidates in history

Historical Git objects show clear signs of avoidable binary growth:

- `output/playwright/.../projects.png`
- template sample images such as `assets/img/3.jpg`, `assets/img/5.jpg`,
  `assets/img/6.jpg`, `assets/img/8.jpg`, `assets/img/9.jpg`, `assets/img/10.jpg`
- `assets/video/tutorial_al_folio.mp4`
- `assets/pdf/2503.04344.pdf`

Why these matter:

- `output/playwright/...` is generated output and should not live in Git
  history.
- Several large template images are referenced by example project pages in
  `_projects/*.md`, not by your core personal content.
- `assets/pdf/2503.04344.pdf` exists in history but is no longer present in the
  current working tree.

## 4. How to inspect the repository again

Run:

```bash
bin/audit-large-files.sh
```

Optional custom thresholds:

```bash
bin/audit-large-files.sh 1 5
```

This means:

- show current tracked files larger than `1 MiB`
- show historical blobs larger than `5 MiB`

## 5. Recommended workflow for PDFs

For paper PDFs and posters:

1. Keep local copies only when the site should directly host the file.
2. Avoid committing many tiny revisions of the same PDF.
3. Make your edits locally, export once, and commit only the final version.
4. If a stable external PDF already exists, prefer linking to that URL instead
   of duplicating the binary here.

Examples in this repo:

- `_bibliography/papers.bib` already links `LEDiT` to arXiv by `url/html`, so a
  large local paper PDF is unnecessary there unless you intentionally want local
  hosting.
- For `JSEN3103042.pdf`, keep the local copy only if you want the site to serve
  the PDF directly and reliably.

## 6. When to rewrite history

History rewriting is appropriate only when repository size becomes a real cost:

- clone/pull becomes noticeably slow
- GitHub storage becomes annoying
- large generated files were accidentally committed

Use `git filter-repo` only after deciding exactly which paths should disappear
from history. Rewriting history changes commit hashes and requires a force push.

Typical example:

```bash
git filter-repo --path-glob 'output/playwright/**' --invert-paths
```

Do not run that casually on a shared branch without deciding the exact cleanup
scope first.
