"""Generates the slurm-*.sbatch launchers from templates/*.sbatch.tpl.

Defaults reproduce the historical sbatch files. Bash variables in the
templates (``$SLURM_ARRAY_TASK_ID``, ``$HOME``, ...) are left untouched;
only ``${workers}``, ``${cpus}``, ``${mem}``, ``${time}`` and
``${extra_sbatch}`` (account/partition) are substituted.

Examples:
    python tool-generate-slurm.py                      # all stages, defaults
    python tool-generate-slurm.py download --workers 20
    python tool-generate-slurm.py processing --cpus 40 --workers 32 \
        --account myaccount --partition cpu_long --time 2-00:00:00
"""
import argparse
import sys
from pathlib import Path
from string import Template

TEMPLATE_DIR = Path(__file__).parent / "templates"

# "workers" means: download = SLURM array size (--max), converters =
# CONVERT_WORKERS (process pool size). The processing stage takes cpus/mem at
# submit time instead (CPUS/RAM env + slurm-process-wrapper.sh), so only its
# time limit is templated.
STAGES = {
    "download": {"cpus": 1, "workers": 6, "mem": "2G", "time": "7-00:00:00"},
    "processing": {"cpus": None, "workers": None, "mem": None, "time": "7-00:00:00"},
    "convert-json": {"cpus": 15, "workers": 12, "mem": "5G", "time": "1-00:00:00"},
    "convert-tei-txt": {"cpus": 15, "workers": 12, "mem": "5G", "time": "1-00:00:00"},
}


def generate(stage: str, args: argparse.Namespace) -> Path:
    values = dict(STAGES[stage])
    for key in ("cpus", "workers", "mem", "time"):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)

    extra = ""
    if args.account:
        extra += f"\n#SBATCH --account={args.account}"
    if args.partition:
        extra += f"\n#SBATCH --partition={args.partition}"
    values["extra_sbatch"] = extra

    template = Template((TEMPLATE_DIR / f"slurm-{stage}.sbatch.tpl").read_text())
    out_path = Path(args.output_dir) / f"slurm-{stage}.sbatch"
    out_path.write_text(template.safe_substitute(values))
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "stages", nargs="*", choices=[[], *STAGES],
        help="Stages to generate (default: all)",
    )
    parser.add_argument("--cpus", type=int, help="--cpus-per-task (default per stage: "
                        + ", ".join(f"{s}={v['cpus']}" for s, v in STAGES.items()) + ")")
    parser.add_argument("--workers", type=int, help="Worker count (default per stage: "
                        + ", ".join(f"{s}={v['workers']}" for s, v in STAGES.items()) + ")")
    parser.add_argument("--mem", help="--mem (default per stage: "
                        + ", ".join(f"{s}={v['mem']}" for s, v in STAGES.items()) + ")")
    parser.add_argument("--time", help="--time (default per stage: "
                        + ", ".join(f"{s}={v['time']}" for s, v in STAGES.items()) + ")")
    parser.add_argument("--account", help="--account (default: none)")
    parser.add_argument("--partition", help="--partition (default: none)")
    parser.add_argument("--output-dir", default=".", help="Where to write the .sbatch files")
    args = parser.parse_args()

    stages = args.stages or list(STAGES)
    overridden = [k for k in ("cpus", "workers", "mem", "time") if getattr(args, k) is not None]
    if overridden and len(stages) > 1:
        print(f"[WARNING] --{', --'.join(overridden)} will apply to ALL of: {', '.join(stages)}",
              file=sys.stderr)
    for stage in stages:
        print(f"Wrote {generate(stage, args)}")
