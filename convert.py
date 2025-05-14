import glob
import os.path
from typing import List, Union, Optional
import tqdm
import re
import lxml.etree as et
import dataclasses
import tarfile
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed


_RUBRICATED = {"rend": "rubricated"}
_INCIPIT = {"rend": "rubricated", "type": "incipit"}


@dataclasses.dataclass
class Line:
    type_: str
    content: str

    @property
    def type(self):
        if self.type_ == "":
            return "fw"
        return self.type_

    @property
    def text(self):
        return self.content + "\n"

    def tei(self) -> et.ElementBase:
        if self.type == "HeadingLine":
            tag = et.Element("span", attrib=_RUBRICATED)
            subtag = et.Element("lb")
            subtag.tail = self.text
            tag.append(subtag)
        elif self.type == "InterlinearLine":
            tag = et.Element("note", attrib={"type": "interlinear"})
            tag.text = self.text
        elif self.type == "DefaultLine":
            tag = et.Element("lb")
            tag.tail = self.text
        elif self.type == "DropCapitalLine":
            tag = et.Element("lb", attrib={"type": "drop-capital"})
            tag.tail = self.text
        elif self.type == "fw":
            tag = et.Element("fw", attrib={"type": "line"})
            tag.text = self.text
        else:
            # print(self.type)
            tag = et.Element("fw", attrib={"type": "line", "subtype": self.type})
            tag.text = self.text
        if tag.tail is None or not tag.tail.strip():
            tag.tail = "\n"
        return tag

@dataclasses.dataclass
class Zone:
    type_: str
    content: List[Line] = dataclasses.field(default_factory=list)

    @property
    def type(self):
        if self.type_ == "":
            return "fw"
        return self.type_

    def tei(self) -> Union[et.ElementBase, List[et.ElementBase]]:
        if self.type == "fw":
            tag = et.Element("fw")
        elif self.type == "RunningTitleZone":
            tag = et.Element("fw", attrib={"type": "running-title"})
        elif self.type == "NumberingZone":
            tag = et.Element("fw", attrib={"type": "numbering"})
        elif self.type == "QuireMarksZone":
            tag = et.Element("fw", attrib={"type": "quire-mark"})
        elif self.type == "MarginTextZone":
            tag = et.Element("note", attrib={"type": "marginal"})
        elif self.type == "MainZone":
            tag = []
        elif self.type == "DropCapitalZone":
            tag = []
        else:
            tag = []
        for line in self.content:
            tag.append(line.tei())
        return tag

XSL = et.XSLT(et.parse("assets/01-simplify.xsl"))
XSL2 = et.XSLT(et.parse("assets/02-raw-text.xsl"))


def get_pages_in_sequence(directory: str):
    mets = et.parse(f"{directory}/METS.xml")
    pages = mets.xpath("/m:mets/m:fileSec/m:fileGrp/m:file/m:FLocat/@x:href", namespaces={
        "m": "http://www.loc.gov/METS/",
        "x": "http://www.w3.org/1999/xlink"
    })
    return list(map(lambda x: f"{directory}/{x}", pages))


def simplify_and_lines(tar_gz, alto_path: str) -> List[Zone]:
    try:
        with tarfile.open(tar_gz, 'r:gz') as tar:
            member = tar.getmember(alto_path.replace(".jpg", ".xml"))
            with tar.extractfile(member) as f:
                xml = et.parse(f)
    except Exception as E:
        print(tar_gz, alto_path, E)
        return []
    xml = XSL(xml)
    zones: List[Zone] = []
    for zone in xml.xpath("//region"):
        zones.append(Zone(type_=zone.attrib["type"]))
        for line in zone.xpath("./line"):
            if line.text and line.text.strip():
                zones[-1].content.append(
                    Line(
                        type_=str(line.attrib["type"]),
                        content=str(line.text).strip()
                    )
                )
        if len(zones[-1].content) == 0:
            zones.pop(-1)
    return zones


def to_tei(file_order: List[str], tar_gz: str, output):
    body = et.Element("body")
    div = et.Element("div", attrib={"n": "pr"})
    body.append(div)
    ab = et.Element("ab")
    div.append(ab)
    last_elem = ab
    for page in file_order:
        zones = simplify_and_lines(tar_gz, page)
        last_elem.append(et.Element("pb", attrib={"n" : os.path.basename(page).replace(".xml", "")}))
        for zone in zones:
            z = zone.tei()
            if isinstance(z, list):
                for line in z:
                    last_elem = ab
                    if line.tag != "lb" and ab.getchildren():
                        last_line = ab.getchildren()[-1]
                        if last_line.tag == line.tag and last_line.attrib == line.attrib:
                            for child in line.getchildren():
                                last_line.append(child)
                        else:
                            ab.append(line)
                    else:
                        ab.append(line)
            else:
                ab.append(z)
    with open(f"{output}/{os.path.basename(tar_gz)}-tei.xml", "w") as f:
        f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<?xml-model href="http://www.tei-c.org/release/xml/tei/custom/schema/relaxng/tei_all.rng" type="application/xml" schematypens="http://relaxng.org/ns/structure/1.0"?>
<?xml-model href="http://www.tei-c.org/release/xml/tei/custom/schema/relaxng/tei_all.rng" type="application/xml"
	schematypens="http://purl.oclc.org/dsdl/schematron"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
      <fileDesc>
         <titleStmt>
            <title>{os.path.basename(tar_gz)}</title>
         </titleStmt>
         <publicationStmt>
            <p>Publication Information</p>
         </publicationStmt>
         <sourceDesc>
            <p>Information about the source</p>
         </sourceDesc>
      </fileDesc>
     <encodingDesc>
        <refsDecl>
           <citeStructure match="//body/div" use="@n">
              <citeData use="head[@type='modern']" property="dc:title"/>
              <citeData use="head[@type!='']" property="dc:title"/>
           </citeStructure>
        </refsDecl>
     </encodingDesc>
  </teiHeader>
  <text>
      {et.tostring(body, encoding=str, pretty_print=True)}
  </text>
</TEI>
""")
        return f"{output}/{os.path.basename(tar_gz)}-tei.xml"


def get_manifest_and_xmls(tar_gz_path: str) -> Tuple[str | None, List[str]]:
    """
    Extracts the path to 'manifest.txt' and a list of all other .xml files in a .tar.gz archive.

    Returns:
        A tuple: (manifest_path, list_of_xml_paths)
    """
    manifest = None
    xml_files = []

    with tarfile.open(tar_gz_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.isfile():
                if member.name.endswith('manifest.txt'):
                    manifest = member.name
                elif member.name.endswith('.xml'):
                    xml_files.append(member.name)

    return manifest, xml_files


def read_file_from_tar(tar_gz_path: str, file_name: str) -> str:
    """
    Reads the content of 'manifest.txt' from a .tar.gz archive.

    Args:
        tar_gz_path: Path to the .tar.gz archive.
        manifest_name: The path to the manifest file inside the archive.

    Returns:
        The content of the manifest file as a string.
    """
    with tarfile.open(tar_gz_path, 'r:gz') as tar:
        member = tar.getmember(file_name)
        with tar.extractfile(member) as f:
            return f.read().decode('utf-8')


def find_tar_gz_files_recursive(dir_path: str) -> List[str]:
    """
    Explicitly recursive version to find all .tar.gz files in a directory tree.

    Args:
        dir_path (str): Root directory path to start the search.

    Returns:
        List[str]: A list of paths to found .tar.gz files.
    """
    matches = []
    for entry in os.scandir(dir_path):
        if entry.is_dir(follow_symlinks=False):
            matches.extend(find_tar_gz_files_recursive(entry.path))
        elif entry.is_file() and entry.name.endswith(".tar.gz"):
            matches.append(entry.path)
    return matches


# Assumes these functions are already defined:
# get_manifest_and_xmls, read_file_from_tar, to_tei, XSL2


def process_tar(file: str):
    if len(glob.glob(f"txt/*/{os.path.basename(file)}.txt")):
        return file, "success"
    try:
        batch = f"batch-{len(glob.glob('txt/*')) // 1000:03d}"
        os.makedirs(f"tei/{batch}", exist_ok=True)
        os.makedirs(f"txt/{batch}", exist_ok=True)

        _, xmls = get_manifest_and_xmls(file)
        manifest = read_file_from_tar(file, "manifest.txt")
        manifest_uri, *files = manifest.split("\n")
        files = [os.path.basename(f) for f in files]
        filepath = to_tei(files, file, f"tei/{batch}")
        conversion = XSL2(et.parse(filepath))
        output_path = f"txt/{batch}/{os.path.basename(file)}.txt"
        with open(output_path, "w") as f:
            f.write(str(conversion))
        return file, "success"
    except Exception as e:
        return file, f"error: {e}"


if __name__ == "__main__":
    files = find_tar_gz_files_recursive("targz")
    l = len(files)
    # files = [
    #     f for f in files
    #     if not re.findall(r"-decor|-filigran|-reliur", f)
    # ]
    # print(f"Removed {l-len(files)} non complete manuscript")
    results = []
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(process_tar, file): file for file in files}
        with tqdm.tqdm(total=len(futures)) as bar:
            for future in as_completed(futures):
                file, status = future.result()
                results.append((file, status))
                bar.update(1)

    # Optional: Print or log errors
    for file, status in results:
        if status != "success":
            print(f"{file} failed with {status}")