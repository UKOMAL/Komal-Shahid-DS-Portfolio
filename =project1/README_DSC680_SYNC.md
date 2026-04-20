# Mental health project — Extra Course ↔ DSC680

**`Extra Course/=project1/project1_mental_health/`** is a **working mirror** of:

**`DSC680/project1_mental_health/`** (the copy that is committed and pushed to GitHub from the `DSC680` repo).

## How to use

1. **Refresh this folder from the portfolio repo** (after you pull or edit in `DSC680`):

```bash
rsync -a --delete \
  --exclude='.DS_Store' \
  "/Users/komalshahid/Desktop/Bellevue University/DSC680/project1_mental_health/" \
  "/Users/komalshahid/Desktop/Bellevue University/Extra Course/=project1/project1_mental_health/"
```

2. **Push changes from Extra Course back into `DSC680`** (when you edited here first):

```bash
rsync -a --delete \
  --exclude='.DS_Store' \
  "/Users/komalshahid/Desktop/Bellevue University/Extra Course/=project1/project1_mental_health/" \
  "/Users/komalshahid/Desktop/Bellevue University/DSC680/project1_mental_health/"
```

Then in `DSC680`: `git add`, `git commit`, `git push` as usual.

## Notes

- In **Extra Course**, the **`project1_mental_health/`** folder is listed in **`.gitignore`** so it stays a **local rsync mirror only** — the version you push to GitHub lives under **`DSC680`**, not here (avoids maintaining two copies in git).
- **`discussions/`** is gitignored inside `DSC680` (private drafts); your mirror here can still hold those files locally for your own use.
- The older **`docs/`** tree under `=project1/` is the pre-reorg layout; **`project1_mental_health/`** matches the portfolio on GitHub (milestones, code, figures).
