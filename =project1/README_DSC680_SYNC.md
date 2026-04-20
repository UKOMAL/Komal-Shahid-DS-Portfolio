# DSC680 Project 1 — Extra Course ↔ DSC680

**`Extra Course/=project1/project1-dsc680/`** is a **working mirror** of:

**`DSC680/projects/project1-dsc680/`** (committed and pushed to GitHub from the `DSC680` repo).

## How to use

1. **Refresh this folder from the portfolio repo** (after you pull or edit in `DSC680`):

```bash
rsync -a --delete \
  --exclude='.DS_Store' \
  "/Users/komalshahid/Desktop/Bellevue University/DSC680/projects/project1-dsc680/" \
  "/Users/komalshahid/Desktop/Bellevue University/Extra Course/=project1/project1-dsc680/"
```

2. **Push changes from Extra Course back into `DSC680`** (when you edited here first):

```bash
rsync -a --delete \
  --exclude='.DS_Store' \
  "/Users/komalshahid/Desktop/Bellevue University/Extra Course/=project1/project1-dsc680/" \
  "/Users/komalshahid/Desktop/Bellevue University/DSC680/projects/project1-dsc680/"
```

Then in `DSC680`: `git add`, `git commit`, `git push` as usual.

## Notes

- In **Extra Course**, **`=project1/project1-dsc680/`** is typically **gitignored** so it stays a **local rsync mirror** — the canonical copy is under **`DSC680/projects/`** on GitHub.
- **`discussions/`** is gitignored in `DSC680` (private drafts); your mirror can still hold those files locally.
- The older **`docs/`** tree under `=project1/` is legacy; **`project1-dsc680/`** matches the portfolio layout (milestones, code, figures).
