# third_party

Externally-authored assets used in experiments. To keep the git history light,
**binaries are not committed** — each asset folder has a `fetch.sh` that
downloads from upstream and verifies SHA-256 checksums.

## Layout

```
third_party/
  splats/
    <source>-<slug>-<id>/
      SOURCE.md     # attribution, license, upstream URL
      fetch.sh      # downloads files + verifies SHA-256
      .gitignore    # ignores the downloaded binaries
```

## Materializing the assets

```sh
# one scene
third_party/splats/superspl.at-woods-3639ecf9/fetch.sh

# everything
third_party/splats/fetch-all.sh
```

Re-running is idempotent: existing files with a matching SHA-256 are skipped.

## Licensing

Only redistribute assets whose license permits it. Each `SOURCE.md` records
the upstream license note verbatim and the required attribution line. The
exact Creative Commons variant (CC0 / CC-BY / CC-BY-SA / …) shown in
SuperSplat's download dialog should be confirmed before redistribution.
