CoMMA Downloading Pipeline
==========================

## Installation

In order to use this pipeline, you need to run `pip install -r requirements.txt` or create an environment at ./env

## Running

The launchers in `slurm/` are generated (and git-ignored) from `templates/*.sbatch.tpl`:

```bash
python scripts/tool-generate-slurm.py                 # all stages, with the default values
python scripts/tool-generate-slurm.py download --workers 20
python scripts/tool-generate-slurm.py processing --cpus 40 --workers 32 --account foo --partition cpu_long --time 2-00:00:00
```

`--cpus`, `--workers` (download: SLURM array size; processing: `KRAKEN_BATCH_SIZE`; converters: process-pool size), `--mem`, `--time`, `--account` and `--partition` are all optional and default to the values used in production.

Run this pipeline by launching two parallel tasks, one running the `slurm/slurm-processing.sbatch` (Check the model used) and one the `slurm/slurm-download.sbatch` (Needs a list of IIIF manifests).
 You can change the file it uses to run download, it basically needs a CSV with a `manifest_url` column.

You will need to change the models configuration

## Pipeline stages

1. **Download** — `worker_single_download.py` (via `slurm/slurm-download.sbatch`, SLURM array): downloads IIIF manifests as CSVs into `output/`, then the images into one kebab-cased directory per manuscript (with a `.manifest.json` tracking file) under `data-in-process/` (override with `DATA_DIR`; these directories are transient — deleted once archived — and directories left at the repo root by older runs are still processed).
2. **Process** — `worker_process.py` (via `slurm/slurm-processing.sbatch`): watches for downloaded images, runs YOLO layout segmentation (`yolalto`) then Kraken OCR into ALTO XML, and archives each completed manuscript as `targz/batch_NNN/<name>.tar.gz`. Each archive embeds its provenance (`manifest.json` member: manifest URI, per-image source URLs, page order) and is registered in `targz/index.csv`, the authoritative manifest-URI ⇄ archive map.
3. **Convert** — `worker_convert_tei_txt.py` (archives → TEI in `tei/` + plain text in `txt/`) and `worker_convert_json.py` (archives → JSON with layout geometry and per-page image URLs in `json/`).

Utilities: `scripts/stats-count-pages-from-archives.py` (page counts), `scripts/bugfix-check-undownloaded.py` (finds missing images), `scripts/tool-rebuild-targz-index.py` (backfills `targz/index.csv` for archives made by older pipeline versions — idempotent, run once after upgrading).

See `docs/writing-a-processing-task.md` to add a new processing step (like the YOLO or Kraken tasks).

## Cite

CoMMA Paper
