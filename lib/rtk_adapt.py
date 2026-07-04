from typing import Optional, List, Callable, Tuple
from rtk.task import Task, InputType, InputListType, _sbmsg
from concurrent.futures.thread import ThreadPoolExecutor
import subprocess
import tqdm
import os
import signal
from rtk import utils
from itertools import repeat
import re
from pathlib import Path
import dataclasses
import json
import glob


@dataclasses.dataclass
class Manifest:
    manifest_id: str = ""
    directory: str = ""
    image_order: List[str] = dataclasses.field(default_factory=list)
    total_images: int = 0
    errors: List[str] = dataclasses.field(default_factory=list)
    uris: List[str] = dataclasses.field(default_factory=list)

    def mark(self):
        with open(str(Path(self.directory) / ".mark"), "w") as f:
            f.write(self.manifest_id.strip())

    def check_mark(self) -> bool:
        if os.path.exists(str(Path(self.directory) / ".mark")):
            with open(str(Path(self.directory) / ".mark")) as f:
                m = f.read().strip()
            return m == self.manifest_id
        return False

    @property
    def json_path(self) -> str:
        return str(Path(self.directory) / ".manifest.json")

    @property
    def images(self):
        return self.image_order

    def to_json(self):
        os.makedirs(self.directory, exist_ok=True)
        with open(Path(self.directory) / ".manifest.json", "w") as f:
            json.dump(dataclasses.asdict(self), f)

    def add_errors(self, uri):
        self.errors.append(uri)

    @classmethod
    def from_json(cls, path: str):
        with open(path) as f:
            d = json.load(f)
        return cls(**d)

    @classmethod
    def from_csv( cls, manifest_uri: str, images: List[Tuple[str, str, str]], tracker: Optional["Tracker"] = None):
        # Compute how many images we have to do per manuscript
        image_count = len(images)
        directory = None
        image_order: List[str] = []
        image_uris: List[str] = []
        # We still record stuff in the tracker, just in case
        for (uri, output_directory, filename) in images:
            directory = output_directory
            image_order.append(filename)
            image_uris.append(uri)
            if tracker:
                tracker.register_dir(manifest_uri, output_directory)
                tracker.record_image_order(manifest_uri, filename)

        if tracker:
            # Then register this expectation
            tracker.add_expected(manifest_uri, image_count)

        return cls(
            manifest_id=manifest_uri,
            directory=directory,
            image_order=image_order,
            total_images=image_count,
            uris=image_uris
        )

    def found_images(self):
        return glob.glob(str(Path(self.directory) / "*.jpg"))

    def is_complete(self, checking_function: Optional[Callable[[str], None]] = None, log: bool = False) -> bool:
        if checking_function is None:
            checking_function = lambda x: utils.check_parsable(x) and utils.check_content(x, ratio=1)

        done = [
            file.stem
            for file in Path(self.directory).glob("*.xml")
            if checking_function(str(file))
        ]
        if log:
            print(f"{self.directory}: {len(done)} imgs / {self.total_images} jpgs ({len(self.found_images())}) [{self.manifest_id}]")

        return self.total_images == len(set(done).intersection(self.image_order))


class YaltoCommand(Task):
    """ Runs a Kraken Like command (Kraken, YALTAi)

    KrakenLikeCommand expect `$out` in its command
    """

    def __init__(
            self,
            *args,
            model_path: str,
            binary: str = "yaltai",  # Environment can be env/bin/yaltai
            output_format: Optional[str] = "xml",
            desc: Optional[str] = "yolalto",
            check_content: bool = False,
            batch_size: int = 1,
            max_time_per_op: int = 60,  # Seconds
            **kwargs):
        super(YaltoCommand, self).__init__(*args, **kwargs)
        self._output_format: str = output_format
        self.check_content: bool = check_content
        self._output_files: List[str] = []
        self.max_time_per_op: int = max_time_per_op
        self.desc: str = desc
        self.model_path = model_path
        self.allow_failure = kwargs.get("allow_failure", True)

        self.command = f"{binary} --batch-size {batch_size} R {self.model_path}".split(" ")
        self.command: List[str] = [x for x in self.command if x]

    def rename(self, inp):
        return os.path.splitext(inp)[0] + ".xml"

    @property
    def output_files(self) -> List[InputType]:
        return list([
            self.rename(file)
            for file in self._output_files
        ])

    @staticmethod
    def pbar_parsing(input_string: str) -> List[str]:
        x = [match for match in re.findall(r"saving to (.*)", input_string.strip())]
        return x

    def check(self) -> bool:
        all_done: bool = True
        for inp in tqdm.tqdm(
                self.input_files,
                desc=_sbmsg("Checking prior processed documents"),
                total=len(self.input_files)
        ):
            out = self.rename(inp)
            if os.path.exists(out):
                if isinstance(self.check_content, bool):
                    self._checked_files[inp] = not self.check_content or utils.check_content(out)
                else:
                    self._checked_files[inp] = self.check_content(out)
            else:
                self._checked_files[inp] = False
                all_done = False
        self._output_files.extend([self.rename(inp) for inp, status in self._checked_files.items() if status])
        return all_done

    def _process(self, inputs: InputListType) -> bool:
        """ Use parallel """

        def work(input_list: List[str], pbar) -> List[str]:
            cmd = []
            for x in self.command:
                if x != "R":
                    cmd.append(x)
                else:
                    cmd.extend(input_list)

            # This allows to control the number of threads used in a subprocess
            my_env = os.environ.copy()
            # my_env["OMP_NUM_THREADS"] = "1"
            # The following values are necessary for parsing output
            my_env["LINES"] = "40"
            my_env["COLUMNS"] = "300"

            out = []

            proc = subprocess.Popen(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=my_env,
                preexec_fn=lambda: signal.alarm(len(input_list) * self.max_time_per_op),
            )

            try:
                for line in iter(proc.stdout.readline, ""):
                    #print(line)
                    for element in self.pbar_parsing(line):
                        out.append(element)
                        pbar.update(1)
                        break
                    if len(set(out)) == len(set(input_list)):
                        continue

                return_code = proc.wait()

                if proc.returncode == 1:
                    print("Error detected in subprocess...")
                    print(proc.stdout.read())
                    print(proc.stderr.read())
                    print("Stopped process")
                    if not self.allow_failure:
                        raise InterruptedError
            except subprocess.TimeoutExpired as te:
                try:
                    print(proc.stderr.read())
                    proc.kill()
                except Exception as E:
                    return out
                return out
            return out

        # Group inputs into the number of workers
        total_texts = len(inputs)
        inputs = utils.split_batches(inputs, self.workers)

        tp = ThreadPoolExecutor(len([batches for batches in inputs if len(batches)]))
        bar = tqdm.tqdm(desc=_sbmsg(f"Processing {self.desc} command"), total=total_texts)
        for gen in tp.map(work, inputs, repeat(bar)):
            for elem in gen:
                if isinstance(elem, str):
                    self._output_files.append(elem)
        bar.close()
