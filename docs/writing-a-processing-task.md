# Writing a processing task

The processing worker (`worker_process.py`) is a chain of **rtk tasks**: each
stage takes a list of input files, produces one output file per input, and can
be re-run safely (already-done inputs are skipped). The current stages are
YOLO layout segmentation (`YaltoCommand`, `lib/rtk_adapt.py`) and Kraken OCR
(`KrakenDirectTask`, `lib/direct_kraken.py`). This document explains the
contract and the two implementation patterns so you can add your own stage.

## The `rtk.task.Task` contract

A task subclasses `rtk.task.Task` and provides:

```python
class MyTask(Task):
    def __init__(self, *args, check_content=False, **kwargs):
        super().__init__(*args, **kwargs)      # first positional arg = list of input files
        self.check_content = check_content     # how to validate an output file
        self._output_files: List[str] = []     # inputs successfully processed

    def rename(self, inp: str) -> str:
        """Map an input path to its output path (e.g. .jpg → .xml)."""

    @property
    def output_files(self) -> List[str]:
        return [self.rename(f) for f in self._output_files]

    def check(self) -> bool:
        """Mark already-done inputs. Called by process() before _process()."""

    def _process(self, inputs) -> bool:
        """Do the actual work on the not-yet-done inputs."""
```

Callers only invoke `task.process()` (defined by rtk): it runs `check()`, and
if anything is left to do, calls `_process()` with the remaining inputs.

### `check()` — resumability

This is what makes the pipeline restartable after a SLURM timeout. For every
input, decide whether its output already exists **and is valid**, and record
that in `self._checked_files[inp] = bool`. Also extend `self._output_files`
with the inputs that were already done, so `output_files` reports them.
Return `True` only if everything was already done.

`check_content` conventions differ between the two existing tasks:

- `YaltoCommand`: a `Callable[[str], bool]` (or bool) applied to the *output*
  path — see `YaltoCommand.check()` at `lib/rtk_adapt.py:105`.
- `KrakenDirectTask`: a `Callable[[str], Tuple[str, bool]]` returning
  `(input_path, ok)`, because checks run in a thread pool and results must be
  matched back to their input — see `lib/direct_kraken.py:200` and
  `custom_ocr_check_with_inp` in `worker_process.py`.

### `_process(inputs)` — the work

Two proven patterns:

**Pattern A — subprocess wrapper (`YaltoCommand`, `lib/rtk_adapt.py:61`).**
Use when the tool is a CLI (here `yolalto`). Key points:

- Build the command once; the placeholder `R` is replaced by the input batch.
- `subprocess.Popen` with `preexec_fn=lambda: signal.alarm(n_inputs * max_time_per_op)`
  as a hard timeout — a hung batch kills itself instead of stalling the worker.
- Parse progress from stdout (`pbar_parsing` regexps "saving to <path>") and
  append each reported file to `self._output_files` as it completes, so a
  crash mid-batch still counts the finished files.
- Batches are distributed over a `ThreadPoolExecutor` (threads are fine: the
  work happens in the subprocess).

**Pattern B — in-process pool (`KrakenDirectTask`, `lib/direct_kraken.py:152`).**
Use when calling a Python library directly (here kraken). Key points:

- `ProcessPoolExecutor` (not threads — the work is CPU-bound Python) with
  `workers` processes; one input per `submit`.
- The worker function (`ocr_and_check`) re-checks the output first and returns
  the input path on success, `None`-ish otherwise, so retried batches are cheap.
- `future.result(timeout=...)` plus a per-future `try/except`: one failed page
  must not fail the batch.
- Pin BLAS threads (`OPENBLAS_NUM_THREADS=1` etc., done at the top of
  `worker_process.py`) or every pool process spawns its own thread storm.

## Wiring the task into `worker_process.py`

Stages are chained inside `process_worker()`, with a validation filter between
each stage — never feed unvalidated outputs downstream:

```python
xmls = YaltoCommand(images, binary="yolalto", model_path=..., check_content=custom_layout_check)
xmls.process()
files = [f for f in xmls.output_files if custom_layout_check(f)]   # filter!

kraken = KrakenDirectTask(files, model=..., check_content=custom_ocr_check_with_inp, ...)
kraken.process()
```

To insert a new stage, construct it with the previous stage's filtered
`output_files`, call `.process()`, then filter its `output_files` with your
own `custom_*_check` before the next stage. If your stage produces the final
per-page XML, make sure `custom_ocr_check` (used by `Manifest.is_complete`
for archiving) recognizes your outputs as done.

## Minimal skeleton

```python
import os
from typing import List
from rtk.task import Task, InputListType


class MyStage(Task):
    def __init__(self, *args, check_content=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_content = check_content
        self._output_files: List[str] = []

    def rename(self, inp: str) -> str:
        return os.path.splitext(inp)[0] + ".out.xml"

    @property
    def output_files(self) -> List[str]:
        return [self.rename(f) for f in self._output_files]

    def check(self) -> bool:
        all_done = True
        for inp in self.input_files:
            out = self.rename(inp)
            done = os.path.exists(out) and (not self.check_content or self.check_content(out))
            self._checked_files[inp] = done
            all_done = all_done and done
        self._output_files.extend(i for i, ok in self._checked_files.items() if ok)
        return all_done

    def _process(self, inputs: InputListType) -> bool:
        for inp in inputs:
            try:
                ...  # produce self.rename(inp)
                self._output_files.append(inp)
            except Exception as exc:
                print(f"[MyStage] {inp} failed: {exc}")
        return len(self._output_files) == len(self.input_files)
```

## Checklist

- [ ] `check()` is idempotent and validates content, not just existence —
      truncated files from a killed job must be redone.
- [ ] One bad input never aborts the batch (`try/except` per input, or
      `allow_failure`).
- [ ] Every operation has a timeout (`signal.alarm` for subprocesses,
      `future.result(timeout=...)` for pools).
- [ ] Inputs are never deleted or modified — only the archiver
      (`worker_process.py:archive`) removes directories, after completeness
      is proven.
- [ ] Outputs are validated *again* in `process_worker` before the next stage.
- [ ] Output naming keeps the input stem (`rename` only swaps the extension) —
      archiving matches pages back to `image_order` by stem.
