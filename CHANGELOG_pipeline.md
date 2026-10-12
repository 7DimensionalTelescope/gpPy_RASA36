# RASA36 Pipeline — Changelog

Comparison of the current pipeline `IMSNG_Routine_RASA36.py` (1983 lines) against the
initial release `IMSNG_Routine_RASA36_old.py` (1661 lines).

- **Original author:** Gregory S.H. Paek (2023.04.24)
- **Intermediate update:** Sophia Kim (2026.05.08)
- **Major rework:** YoungPyo Hong (2026.05.20) — *"Upgrade Logging / implement
  multiprocessing / delete hardcoded paths / modify it for RASA36"*

The pipeline keeps the same overall gpPy stage sequence:

```
header correction → master frames → object calibration → astrometry →
(CR copy) → rename → SCAMP/MissFITS WCS → photometry → image stack →
photometry (combined) → subtraction → photometry (subtracted) →
transient search → summary → file transfer
```

What changed is *how* each stage runs, where data goes, and how the run is observed.

---

## 1. Scope & Invocation: generic GECKO tool → RASA36-only

| | Old | New |
|---|---|---|
| `obs` | `sys.argv[1]`, prompted if absent | Hardcoded `obs = 'RASA36'` |
| Input | Scanned a raw-data tree (`/data6/obsdata/{obs}/2*`) and auto-selected the next unprocessed night from a log file | An **explicit input folder path** passed as `sys.argv[1]` |
| `ncores` | `argv[2]`, default **1** | `argv[2]`, default **4** |
| Multi-facility logic | KCT / LOAO / LSGT_SNUCAMII special cases throughout | Removed (RASA36-specific; a few `KCT`/`LSGT` branches remain dormant in borrow logic) |

- The old "find new data" mechanism (`logtbl`, `datalist`, `newlist`, `No new data` /
  `sys.exit()`) is **gone**. The new script processes exactly the one folder it is handed,
  which makes it safe for concurrent/manual runs and for being driven by an external
  scheduler.
- `mode`/`badmode` (RASA `high`/`hdr` gain mode, with the ×20 master-frame multiply for
  "bad mode") is retained.

---

## 2. Paths de-hardcoded & output reorganized

**Old** baked absolute, user-specific paths everywhere:
`/data4/gecko/factory`, `/home/paek/log`, `/home/paek/table`, `/data3/paek/factory/master_frames`,
`/data6/bkgdata/{obs}`, `/data6/IMSNG/IMSNGgalaxies`, `/data3/paek/factory/ref_frames/{obs}`.

**New** derives everything from the script location:

```python
HERE = os.path.dirname(os.path.abspath(__file__))
path_result  = '/data6/processed_RASA'                 # the only external output root
path_factory = os.path.normpath(os.path.join(HERE, '..', 'factory'))
path_mframe  = os.path.join(path_factory, 'master_frames')
path_config  = os.path.join(HERE, 'config')
path_astrometry = os.path.join(path_config, 'astrometry_gppy_rasa36.config')
...
```

- Single external output root: **`/data6/processed_RASA`**.
- Dropped the `alltarget.dat` dependency entirely (the IMSNG galaxy-name correction and the
  RA/Dec catalog lookup for astrometry are commented out — astrometry is now purely
  header-based).
- Removed the old `path_log` / `path_save` (`/data6/bkgdata`) machinery and the
  `path_phot_sg` / `path_phot_sub_2021` legacy photometry-script variables that were unused.

---

## 3. Multiprocessing (the largest code change)

The old script was mostly serial; only astrometry used a pool. The new script introduces a
forked context and parallelizes nearly every per-image / per-group stage:

```python
mp_fork = multiprocessing.get_context('fork')
```

Parallelized stages:

| Stage | Mechanism |
|---|---|
| Master **dark** (per exptime) | `mp_fork.Process` per exptime, results read back from disk |
| Master **flat** (per filter) | `mp_fork.Process` per filter, read back from disk |
| **Object calibration** (per filter) | `mp_fork.Process` per filter — single filter runs in-process to avoid fork memory doubling |
| **Astrometry** | `mp_fork.Pool` (constrained → blind → retry passes) |
| **CR copy** (`a*` → `cra*`) | `ThreadPoolExecutor` |
| **Rename** to IMSNG convention | `mp_fork.Pool` over `calib.fnamechange` |
| **Pre-SExtractor** (SCAMP pass A) | `mp_fork.Pool` over `run_pre_sextractor` |
| **WCS keyword update** | `mp_fork.Pool.imap_unordered` |
| **Image stack** | `mp_fork.Pool` over `_stack_group_worker` — whole `(obj, filter, group)` task per worker |
| **Subtraction** | `mp_fork.Pool.imap_unordered` over `_subtract_one` |
| **PNG previews** | `ThreadPoolExecutor` |

- Fork is deliberately chosen so that closures and in-memory `CCDData` objects are inherited
  without pickling. Master frames are still **written to disk by the workers and re-read** by
  the parent (workers mutate fork-inherited state, so the parent reads results back from the
  files rather than from return values).
- Worker exit codes are checked: `if p.exitcode != 0: raise RuntimeError(...)`.

### Calibration bug fix
Object calibration now passes `exptime=expt` into `calib.calibration`:

```python
calib.calibration(ic1, mzero, flatdict[filte], filte, mdark=mdark, exptime=expt)
```

The inline comment documents the bug this fixes: previously *all* images of a filter were
re-processed once per exptime value, with the last dark overwriting earlier results.

### Calibrated frames float64 → float32 (2026-07-08)
`ccdproc.subtract_bias`/`subtract_dark(scale=True)`/`flat_correct` promoted each calibrated
science frame to **float64 (BITPIX -64, 8 B/px)**, so every `fz*/Calib*` frame was ~134 MB
(4096² × 8) — double what it needs to be, doubled again by the MissFITS `.back` backup. The
SWarp coadd is float32, so a *stack* ended up **smaller on disk than its own input frames**
even though it covers a larger (dithered) footprint — a confusing "stacking shrank the file"
symptom. Fix: cast to `float32` before writing in `calib.calibration` (and in `calib.defringe`,
where `data - fscale*fringe` re-promotes to float64). CCD ADUs fit float32 easily; halves
intermediate disk use and makes stack inputs consistent with the coadd.

---

## 4. ToO (Target-of-Opportunity) support — new

Entirely new capability with no equivalent in the old script.

- Per-object metadata is detected up front from `.head` sidecar files (preferred) or, failing
  that, from the FITS headers, into `obj_meta = {object: {is_too, objtype, too_name}}`.
- `_check_too(...)` flags an object as ToO if **any** of:
  1. `IS_TOO` or `is_rapidToO` header is `True`;
  2. `OBJTYPE` header is `Request`;
  3. `"too"` appears in the input folder name or a raw filename.
- ToO images get a `NOTE` FITS keyword carrying the event/folder name.
- Output routing depends on the flag (see §9):
  - ToO → `ToO/{date}/{too_name}/{filter}/`
  - normal → `{objtype}/{obj}/{date}/{filter}/`

New observing-metadata keywords are also preserved through calibration into the stacks:
`OBSERVER`, `NOTES`, `OBJTYPE`, `IS_TOO`, `is_rapidToO`, `OBSID`, `PROGID`.

---

## 5. Logging, error handling & progress

- **Every stage** is wrapped in `try / except Exception as e:` that logs the error plus full
  traceback and re-raises:
  ```python
  except Exception as e:
      logger.error(f'Unexpected error occured: {e}')
      logger.exception('Full error traceback:')
      raise
  ```
- Most `print(...)` calls are replaced with `logger.info / warning / error`.
- `tqdm` progress bars are added across all loops (classification, astrometry passes, SCAMP
  groups, stacking, subtraction, file transfer, PNG generation, etc.).
- The log file path is tracked (`_logfile_path`) so the completed log can be copied into each
  processed output directory at the end.

---

## 6. Slack notifications: start/end → live threaded progress

- Old: a single start message and a single completion message (flat, in-channel).
- New: a start message captures a thread (`slack_ts = tool.slack_bot(...)`), and every stage
  posts a threaded progress update **`[1/9] … [9/9]`** with metrics:
  - **Start:** raw science frame count, filters, exptimes, cores, total data size (MB and
    MB/frame).
  - **[1/9] Master frames**, **[2/9] Pre-process**, **[3/9] Astrometry** (OK/fail + %),
    **[4/9] Photometry**, **[5/9] Stacking**, **[6/9] Photometry (combined)**,
    **[7/9] Subtraction**, **[8/9] Photometry (subtracted)**, **[9/9] Transient search** —
    each with elapsed seconds.
  - **Completion:** start/end timestamps, total minutes, the full
    `raw → calibrated → astrometry → stacked → subtracted` funnel, targets, filters, and
    per-stage times.
- Every Slack call is itself wrapped so a notification failure only warns and never breaks the
  run.

---

## 7. PNG previews — new

`save_fits_as_png()` renders a ZScale-normalized grayscale PNG (via
`astropy.visualization.ZScaleInterval`) for quick visual inspection. Generated (in thread
pools) for **calibrated (non-stacked)** frames (`Calib*0.fits`), **stacked**, and
**subtracted** images, then moved alongside the FITS into the output tree. The non-stacked
previews ride along via the file-transfer `tail=''` catch-all rather than a dedicated `.png`
move.

---

## 8. Combined-image handling & subtraction reference

**Stacking (`_stack_group_worker`)** — same SCAMP → MissFITS → SWarp recipe, but now:
- Effective gain uses the **real gain** from `obsinfo['gain']` rather than a hardcoded `1.5`:
  `effgain = (2/3) * N_combine * gain`.
- Adds an `NCOMBINE` keyword.
- Writes a `.head` sidecar (`hdr.totextfile(...)`) so downstream tools can read the WCS.
- Adds the ToO `NOTE` keyword when applicable.

**SExtractor invocation** changed:
- binary `sex` → `source-extractor`;
- added explicit `-GAIN {gain}`;
- config file `simple.config` → `simple.sex`.

**Subtraction reference selection** — the biggest behavioral change here:
- Old: a static reference library, `/data3/paek/factory/ref_frames/{obs}/Ref*{obj}*{filter}*.fits`,
  with a two-step `subtraction_routine2` → `subtraction_routine` fallback.
- New: uses the **earliest prior TOS stack of the same tile**,
  `{path_result}/TOS/{obj}/*/{filter}/Calib*.com.fits`, **excluding the current night**
  (`/{date_str}/`). Single `subtraction_routine2` (wcsremap + hotpants); the intermediate
  wcsremap reference is cleaned up. Works for both TOS and ToO because `OBJECT` is always the
  tile ID.

> Note: the **transient-search** stage (§ near end) still points `path_refim` at `path_ref`
> (`Ref*{obj}*{filter}*.fits`) for its reference column — that path was made relative but the
> `Ref*` library convention is unchanged there.

---

## 9. File transfer, summary & cleanup

- **Summary:** the old script wrote `obs.summary.log` into the data folder; the new script
  builds a richer `summary_lines` block (timing, calibration counts, the science funnel,
  targets, filters, per-stage times) and logs it.
- **Output layout** is reorganized under `/data6/processed_RASA`:
  - normal: `{objtype}/{obj}/{date}/{filter}/` (+ `/transients`)
  - ToO: `ToO/{date}/{too_name}/{filter}/`
  - `.head` sidecars and `.png` previews are moved alongside the images.
- After transfer, the factory working directory is **deleted** (`shutil.rmtree`), and the run
  log is copied into each processed output directory.
- Old behavior of `mv {path_data} {path_save}` (move the whole working dir to
  `/data6/bkgdata`) and appending to `path_log` is removed.

---

## 10. Input / output folder structure

### Input

| | Old | New |
|---|---|---|
| Location | Scanned a whole tree: `glob('/data6/obsdata/{obs}/2*')` | One **explicit folder** passed as `argv[1]` |
| Selection | Auto-picked the next night not yet in the log | Operator/scheduler hands it the exact folder |
| Folder name | a date string (`20210425`, `20210507_hdr`) | `YYYY-MM-DD_gain25` style, with a `_ToO` suffix for ToO nights |

The new script parses the folder name for context:
- `date_str = basename.split('_')[0]` → e.g. `2026-06-18`
- `mode` from `hdr` / `high` in the name (gain mode)
- `'too' in basename.lower()` is one of the ToO triggers

Real examples on disk:
```
/data6/obsdata/RASA36/image/2026-04-22_gain25
/data6/obsdata/RASA36/image/2026-06-18_gain25_ToO     ← ToO night
```
A raw input folder is flat: all `*.fits/*.fts` frames (bias/dark/flat/object) plus optional
`.head` sidecars, no subdirectories.

### Output

The biggest reorganization. The old script scattered results across two trees keyed by
**galaxy name → observatory**; the new script writes one tree keyed by
**observation type → tile → date → filter**.

**Old layout** (two destinations):
```
# per-target photometry products
/data6/IMSNG/IMSNGgalaxies/{obj}/{obs}/{filter}/
/data6/IMSNG/IMSNGgalaxies/{obj}/{obs}/{filter}/transients/
# the entire working directory, oved wholesale
/data6/bkgdata/{obs}/{night}/
```
m
**New layout** — single root `/data6/processed_RASA`, split by obs type:
```
/data6/processed_RASA/
├── TOS/   {tile}/{date}/{filter}/         ← normal survey (Tiling Observation Strategy)
├── ToO/   {date}/{too_name}/{filter}/     ← target-of-opportunity (date first)
└── Test/
```

Concrete examples:
```
TOS/T00080/2026-04-22/r/Calib-RASA36-T00080-20260423-075947-r-60.fits
                      r/...phot.cat, .zpcal.png, .gcurve.png, .com.fits, .head, ...
                      r/transients/

ToO/2026-06-18/T02094/r/Calib-RASA36-T02094-20260619-072450-r-60.fits
ToO/2026-06-22/T01623/r/...
```

Key differences in the new tree:
- **Keyed by tile ID** (`T00080`), not galaxy name — `OBJECT` is always the tile ID for both
  TOS and ToO.
- **Date level** (`{date}/`) is now part of the path, so repeat visits of the same tile stack
  up as siblings (this is exactly what the subtraction step exploits to find an earlier
  reference of the same tile — see §8).
- **TOS vs ToO are split at the top**, and ToO puts **date before target** (`ToO/{date}/{name}/`)
  whereas TOS puts **target before date** (`TOS/{tile}/{date}/`).
- Each filter leaf holds the **full product set together**: individual + combined (`.com.`)
  FITS, photometry catalogs (`.phot.cat`, `.phot.summary.dat`), zero-point/growth-curve plots,
  `.head` sidecars, PNG previews, a `transients/` subfolder, and a copy of the run log.
- The old "move the entire working dir to `/data6/bkgdata`" step is gone — the factory working
  directory is **deleted** after the curated products are filed into the tree.

---

## Quick reference — new top-level helpers

| Function | Purpose |
|---|---|
| `_read_head(headfile, keyword)` | Parse a single keyword from a SCAMP/MissFITS `.head` sidecar |
| `save_fits_as_png(fitsfile, outpng)` | ZScale PNG preview |
| `run_pre_sextractor(..., gain)` | Now calls `source-extractor` with `-GAIN` |
| `_check_too(...)` / `obj_meta` | ToO detection |
| `_make_master_dark` / `_make_master_flat` | Forked master-frame workers |
| `_calibrate_filter(filte)` | Forked per-filter object calibration |
| `_update_wcs_keywords(calibim)` | Forked RACENT/DECCENT/RAPOLY writer |
| `_stack_group_worker(args)` | Forked per-group stacking unit |
| `_split_group_by_exptime(group, max_sec)` | Chunk a time-gap group by cumulative EXPTIME for `--stack-sec` (2026-07-10) |
| `MAX_SNAPSHOTS` (gregoryfind) | Per-epoch cap (100, top SNR) on transient snapshot stamps; catalog never truncated (2026-07-13) |
| `_subtract_one(inim)` | Forked subtraction unit (TOS-tile reference) |
| `_copy_if_needed` / threaded copies | CR copy and PNG generation |

---

*Generated from a line-by-line comparison of the two scripts.*

---

## 2026-07-13 — Concurrent ToO fast lane

- **`rasa36_watcher.py`**: two pipeline lanes ("main" / "too"), each a single worker
  thread. Tars whose stem ends in `_ToO` (case-insensitive) run in the "too" lane in
  parallel with the regular lane, so a ToO no longer waits behind a nightly reduction.
  Jobs within a lane stay serialized; both lanes use the same `--ncores` (default 4).
- **`IMSNG_Routine_RASA36.py`** concurrency-safety for two simultaneous runs
  (different input folders only — same-folder runs still forbidden):
  - `atomic_cp(src, dst)` — master bias/dark/flat copies into the shared
    `factory/master_frames/` now go via tmp + `os.replace` so a concurrent run never
    reads a half-written master.
  - `scamp_output_opts(tag)` — SCAMP XML + check-plots redirected into the run's
    factory dir with per-group tags (`scamp.A/B.<obj>.<filte>.<gidx>.*`); previously
    they landed in the shared cwd and collided across parallel workers and runs.
  - SWarp `-WEIGHTOUT_NAME`/`-XML_NAME` likewise redirected into the factory dir
    (`weight.<obj>.<filte>.<gidx>.fits`, `swarp.<obj>.<filte>.<gidx>.xml`); previously
    all parallel SWarps overwrote `./coadd.weight.fits` / `./swarp.xml`.
- **Skip-unpack**: `_handle` skips `_extract_tar` when `image/<tar-stem>/` already
  exists and contains FITS files (re-runs of already-extracted tars go straight to the
  pipeline queue); queue/state logic factored into `TarHandler._enqueue`. Delete the
  extracted dir to force re-extraction after a suspected partial unpack.

## 2026-07-13 — Saturated-core frame rejection + raw quarantine

Bright-moon nights (first seen 2026-06-26, MOONPHAS 0.89) can clip the
vignetting-boosted frame center at 16-bit full well; the clipped plateau is
unrecoverable and produced washed-out white-disk coadds with silently-empty
photometry. New gate in `IMSNG_Routine_RASA36.py`, right after header
correction and before master frames:

- Every object frame's central-box median (±`SATCORE_BOX`=400 px) is measured;
  frames at ≥ `SATCORE_LEVEL`=60000 ADU are rejected before any calibration.
- The factory copy is renamed `*.fits.satcore` (excluded from all later globs,
  same convention as `.badwcs`).
- The **original raw files** (`.fits` + `.head` sidecar) are moved out of the
  input folder into `/data6/obsdata/RASA36/fail_img/<YYYY-MM-DD>/` (date folder
  created on demand), so re-runs of the night don't reprocess them.
- `ic1` is rebuilt and `hdr.cor.dat` rewritten after rejection; the Slack start
  message (and the "calibration only" message when all science frames are
  rejected) reports the count.
- Threshold calibrated on 2026-06-26: ruined frames ≥ 63,600 ADU, good frames
  ≤ 45,200 ADU — no overlap.

## 2026-07-13 — Transient search fixed to use the two-tier reference lookup; raw-frame count deduped

- **Transient search reference (bug)**: the transient-search block still resolved its
  reference via the legacy `path_ref` glob (`factory/ref_frames/RASA36/Ref*<obj>*<filter>*`),
  never updated when subtraction got the two-tier lookup (2026-07-08/09). For every tile
  whose template came from TOS or `reference_img/` — i.e. all normal runs — the glob was
  empty, `refimlist[0]` raised `list index out of range`, the `transient_search.txt` row
  stayed blank, and gregoryfind silently produced no candidates despite a successful
  subtraction (seen 2026-06-18_gain25_ToO / T02094). The lookup is now factored into
  `_resolve_refim(obj, filte)` (earliest prior TOS epoch excluding the current date →
  pre-staged `reference_img/` template) and shared by `_subtract_one` and the transient
  search. `_subtract_one` still applies `_prestaged_to_cd` to PRESTAGED refs; a missing
  reference in transient search now raises `FileNotFoundError` (caught + logged per image).
  `factory/ref_frames/` is no longer read by anything; GRB templates only need to be in
  `reference_img/`.
- **`nobj` over-count (reporting)**: `filtered_hdr` filtered `ic1` by imagetyp `'OBJECT'`,
  `'object'`, `'Light Frame'` and `'object'` *again*, then concatenated all matches, so the
  same files were counted up to 3× ("15 raw → 5 calibrated" on a 5-frame night). Duplicate
  filter removed; `filtered_ic1` deduped order-preserving via `dict.fromkeys`.

## 2026-07-13 — Observation summary plots fixed (savefig `overwrite` kwarg)

`tool.obs_summary` failed in every run ("Fail to make summary plots." — 06-18/06-22
ToO, GRB260421B, all runs checked) because its final lines passed `overwrite=True`
to `fig.savefig`; that was never a matplotlib kwarg and current matplotlib raises
`TypeError` where old versions silently ignored it. A bare `except:` in
`IMSNG_Routine_RASA36.py` swallowed the exception, so no `obs.summary.<filter>.png`
was ever produced. Removed the kwarg from all live call sites (`util/tool.py`
`obs_summary` png+pdf and `stampimage`; `config/growthcurve.py`) and the handler now
logs the exception message. Verified by running `obs_summary` on the 2026-06-18
T02094 output — png+pdf render correctly.

## 2026-07-13 — Transient search gated by template quality (subtraction stays universal)

(Replaces a same-day interim design that skipped subtraction entirely for non-ToO
targets via a `--sub-all` flag; that flag no longer exists.)

Subtraction runs for every target as before — it is cheap (~1 min/coadd, parallel)
and its `hd*` images / `phot_sub` catalogs are useful to have on disk. What is now
gated is the **transient search**, per difference image:

- ToO targets (`obj_meta['is_too']`, from `IS_TOO`/`OBJTYPE=Request` headers): always
  searched, regardless of template source.
- TOS tiles: searched only when the subtraction template was a prior TOS epoch
  (`REFSRC='TOS'`). Survey-template (`REFSRC='PRESTAGED'`) subtractions are skipped
  with a log line — those difference images carry ~46% stellar dipole residuals and
  flooded the search with junk (72k candidates / 430k stamps, 2026-07-10).

Mechanics: `_subtract_one` now stamps `REFIM` (template path) and `REFSRC`
(`TOS`/`PRESTAGED`) into the `hd*` header; the transient stage reads them and falls
back to `_resolve_refim` only for hd images predating the stamping. The transient
input table is now built from the kept rows only (previously it was pre-sized and
failed rows were passed to gregoryfind as blank lines). Verified with real headers:
stamping round-trips, ToO images always searched, PRESTAGED-on-TOS skipped,
`transient_search.txt` column set unchanged for `gregoryfind_bulk_mp_2021.py`.

## 2026-07-13 — Stack-stage time grouping dropped the last epoch (fixed)

The stack stage's 1-hour time-gap grouping loop ended with
`elif i == len(delts)-1: grouplists.append(grouplist)` instead of appending the
final group after the loop. Whenever the **last** frame of an (object, filter) set
started a new time group — i.e. it sat >1 h after the previous frame, and **always**
under `--no-stack`, where every frame is its own group — the loop appended the
*previous* group twice (the same list object, so two stack tasks with identical
frame lists) and the final group was never appended:

- The last epoch silently got no `.com.fits` (missing final point in ToO/GRB
  time-series photometry).
- The duplicated group produced two parallel SWarp runs writing the **same**
  `IMAGEOUT_NAME` (the coadd name derives from the combined DATE-OBS/EXPTIME) — a
  write race that can corrupt that coadd, and the duplicate coadd path then flowed
  twice into photometry_com/subtraction (defeating the per-`inim` wcsremap temp
  uniqueness fix of 2026-07-10 for that image).

The SCAMP Pass A copy of the same loop was already correct (append-after-loop, elif
commented out); only the stack copy was wrong. Fixed to match Pass A, verified by
simulating both versions on gap-at-end / --no-stack / contiguous cases (old code:
`[[A],[A]]` + lost `[B]`; new code: `[[A],[B]]`). Nightly runs with one contiguous
visit per tile were unaffected (aliasing made the early append benign). Past
`--no-stack` runs (e.g. GRB260421B) should be checked for a missing final epoch.
Also fixed a missing f-prefix on the stack stage's "IMAGE REGISTERING & COMBINE"
log line (logged the literal `{imlist_tmp}`).

## Per-tile subtraction / transient-search reporting (2026-07-14)

"Subtracted: N/M" alone couldn't distinguish a missing template from a
wcsremap/hotpants failure from a quietly-skipped transient search, so the
subtraction and transient-search stages now report per-tile outcomes in both
the log and their Slack step messages:

- `_subtract_one` returns `(subim, ds9com, obj, filte, ref_src, ref_base)`
  instead of `(subim, ds9com)`/None; the parent aggregates records by
  (tile, filter, outcome, reference) — so a ToO `--no-stack` run with 50
  frames vs one template stays a single line ("(50 images)").
- Log + `[7/9]` Slack message list, per tile: `subtracted, ref [TOS|PRESTAGED]
  <template>`, `NOT subtracted - no reference image`, `FAILED
  (wcsremap/hotpants)`, plus an abandoned-count line if imap_collect timed out.
  Slack detail capped at 25 lines ("… and N more tiles (see log)").
- `[9/9]` Slack message now shows `Searched: N/M (skipped: S, errors: E)` and
  per-tile `searched vs <template>` lines (same 25-line cap); the `cd_` prefix
  of factory CD-converted copies is stripped so the real template is named.
- ToO targets are labeled `"<NOTES name> (<tile>) <filter>"` (e.g.
  `GRB260421B (T02094) r`) in all these lines via `_target_label`.
- `_subtract_one` body is now wrapped in try/except (per-task isolation):
  an unguarded per-frame exception used to abort the whole subtraction stage
  via imap_collect's `.next()` re-raise.

## Empty-catalog crash in gregoryfind + [9/9] outcome audit (2026-07-15)

2026-06-10_gain25's two searchable tiles (T00358/T00359, TOS templates) produced
no `transients/` folder despite Slack reporting `Searched: 2/64 … errors: 0`.
Root cause: both difference images were source-free (`hd*.phot_sub.cat` 0 rows)
and `gregoryfind_bulk_mp_2021.py`'s flag-summary print divided by `n_all=0` —
ZeroDivisionError *before* the `transients.cat` write. `routine_safe` swallowed
it; the traceback went only to gregoryfind stdout, which the reprocess sweep's
conda-run pipe discards; and the pipeline's `errors` counter only covers
building the search table, so the failure was invisible.

- gregoryfind: `n_all == 0` guard writes the (empty) `transients.cat`, prints
  `No sources in difference-image catalog`, and returns — "searched, zero
  candidates" now leaves its catalog on disk (→ `transients/` dir is created),
  distinguishable from a real failure. Verified by re-running the actual
  T00358 inputs in isolation.
- Pipeline transient-search stage: after gregoryfind runs, audits the expected
  `.transients.cat` per searched image. Missing catalog ⇒ counted as a
  gregoryfind failure; existing ⇒ candidates counted (`flag==False` rows).
  Log + `[9/9]` Slack now report `Searched: N/M (skipped: S, errors: E,
  gregoryfind failures: F)`, a `Transient candidates: C` total, and per-tile
  `searched vs <template> — c candidates` lines.

## 2026-07-24 — SCAMP Pass A per-group isolation (one bad tile no longer aborts the night)

Diagnosed from the 2026-07-22 reprocess: 4 of 5 failed folders (2026-06-23, -24,
-25, 2026-07-01) died at the **WCS-refine (SCAMP Pass A)** stage on the *first*
tile group and lost every downstream product. `run_shell_checked(scampcom,
"SCAMP (pass A)")` had no per-group `try/except`, so one tile's SCAMP failure
re-raised through the stage-level handler and killed the whole run — the same
gap the stack stage's `_stack_group_worker` had (fixed 2026-07-15). Two triggers:
exit **139** (SIGSEGV, transient glitched-Gaia-refcat parse — 06-24/25, 07-01)
and exit **1** (a real per-tile SCAMP error — 06-23). Confirmed **not** memory:
~220 GB free, cgroup 50–73 %, pipeline RSS < 2.5 GB at every crash (139 ≠ OOM's 137).

- **Per-group guard** (`IMSNG_Routine_RASA36.py` ~line 1487): the SCAMP + MissFITS
  calls are wrapped in `try/except`. On failure it logs the group + exit code,
  appends `(obj/filte/gN, size)` to a new `passa_skipped` list, and `continue`s
  to the next group instead of raising. A skipped group keeps its solve-field WCS
  and is re-refined by the stack stage's own SCAMP, so it usually self-heals.
- **Slack skip-reporting**: the `[4/10] WCS refine` step message appends a
  `:warning: Skipped N tile group(s) (M frame(s)) on SCAMP/MissFITS failure —
  run continued` line plus the per-group labels (capped at 25, rest in the log)
  when `passa_skipped` is non-empty. A silent whole-run abort is now a visible,
  scoped skip.
- The two SCAMP-Pass-A rows in `CLAUDE.md` (§ Known failure signatures) updated
  accordingly. 06-22 is a **separate** issue (0 stars detected → total astrometry
  failure on a bright-sky night; not addressed by this change — see Fix 2/3 in the
  diagnosis).

## 2026-07-24 — Astrometry total-failure gate (no more silent 0/N "COMPLETE")

Companion to the same-day SCAMP Pass A fix, from the same 2026-07-22 reprocess
diagnosis. Folder 2026-06-22 solved **0 of 427** science frames — SExtractor
found `Objects: detected 0` on every frame across all declinations, because a bad
own-flat on a bright high-sky night (median 30800 ADU, moon 0.57) wiped the stars
(06-23 with a *borrowed* flat solved fine). The run nonetheless proceeded through
empty stack/subtraction and posted a clean `Pipeline Complete` — a silent total
failure.

- **Gate** (`IMSNG_Routine_RASA36.py`, end of the astrometry stage): if
  `n_astro_ok == 0` **and** `n_calib_images >= MIN_FRAMES_FOR_ASTRO_GATE` (20), the
  pipeline logs `CRITICAL - ASTROMETRY TOTAL FAILURE`, posts a `:rotating_light:`
  Slack alert (`Astrometry TOTAL failure - <folder>` + `0/N solved`), and
  `sys.exit(1)`. `SystemExit` is not caught by the stage's `except Exception`, so it
  terminates cleanly. The `[3/10] Astrometry done` step message still posts first, so
  the `0/N` context is in the thread before the alert.
- **Conservative on purpose:** requires ≥20 calibrated frames (a handful of failed
  pointing-test/EXPTEST frames won't trip it) and only fires on *exactly* 0 solves —
  any partial success (≥1 frame) proceeds normally.
- **Re-queue semantics:** the non-zero exit leaves the run unmarked in
  `rasa36_watcher_processed.txt`, so reprocess/watcher re-run it after the underlying
  cause (flat / sky) is addressed — a re-triageable failure instead of a false success.
- Does **not** fix 06-22's root cause (the bad flat) — that's the still-optional
  Fix 3 (flat sanity-check + neighbor-night fallback). This gate only makes the
  failure loud and non-silent.
