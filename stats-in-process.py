import os
import json
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

def analyze_directory(path: Path):
    """Analyze one directory containing .manifest.json, JPGs, and XMLs."""
    manifest_path = path / ".manifest.json"
    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text())
        image_order = manifest.get("image_order", [])
        expected_count = len(image_order)
    except Exception as e:
        console.print(f"[red]Error reading {manifest_path}: {e}[/red]")
        return None

    jpgs = {f.stem for f in path.glob("*.jpg")}
    xmls = {f.stem for f in path.glob("*.xml")}

    present_jpg_count = len(jpgs)
    matching_xml_count = sum(1 for j in jpgs if j in xmls)

    pct_jpg = (present_jpg_count / expected_count * 100) if expected_count else 0
    pct_xml = (matching_xml_count / present_jpg_count * 100) if present_jpg_count else 0
    complete = expected_count == present_jpg_count == matching_xml_count

    return {
        "folder": path.name,
        "expected": expected_count,
        "jpg": present_jpg_count,
        "xml": matching_xml_count,
        "pct_jpg": pct_jpg,
        "pct_xml": pct_xml,
        "complete": complete,
    }

def main(root: str):
    root_path = Path(root)
    dirs = [d for d in root_path.iterdir() if d.is_dir()]
    results = []

    console.print(f"\n[bold cyan]Analyzing {len(dirs)} directories under {root_path}...[/bold cyan]\n")

    for d in tqdm(dirs, desc="Scanning folders"):
        res = analyze_directory(d)
        if res:
            results.append(res)

    # Build pretty table
    table = Table(title="Directory Status Report", show_lines=True)
    table.add_column("Folder", style="bold")
    table.add_column("Expected", justify="right")
    table.add_column("JPGs", justify="right")
    table.add_column("XMLs", justify="right")
    table.add_column("% JPG", justify="right")
    table.add_column("% XML (of JPG)", justify="right")
    table.add_column("Complete", justify="center")

    total_expected = total_jpg = total_xml = complete_count = 0

    for r in results:
        color = "green" if r["complete"] else "yellow" if r["pct_jpg"] > 50 else "red"
        table.add_row(
            r["folder"],
            str(r["expected"]),
            str(r["jpg"]),
            str(r["xml"]),
            f"[{color}]{r['pct_jpg']:.1f}%[/]",
            f"[{color}]{r['pct_xml']:.1f}%[/]",
            "[bold green]✔[/]" if r["complete"] else "[red]✘[/]",
        )

        total_expected += r["expected"]
        total_jpg += r["jpg"]
        total_xml += r["xml"]
        if r["complete"]:
            complete_count += 1

    console.print(table)

    overall_complete_pct = complete_count / len(results) * 100 if results else 0
    overall_jpg_pct = total_jpg / total_expected * 100 if total_expected else 0
    overall_xml_pct = total_xml / total_jpg * 100 if total_jpg else 0

    console.print("\n[bold underline cyan]Global Summary[/bold underline cyan]")
    console.print(f"Total folders analyzed: [bold]{len(results)}[/bold]")
    console.print(f"Complete folders: [green]{complete_count}[/green] ({overall_complete_pct:.1f}%)")
    console.print(f"Total expected images: {total_expected}")
    console.print(f"Total present JPGs: {total_jpg} ({overall_jpg_pct:.1f}%)")
    console.print(f"Total XMLs: {total_xml} ({overall_xml_pct:.1f}% of JPGs)\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze completeness of folders with manifests, JPGs, and XMLs.")
    parser.add_argument("root", help="Root directory containing the subfolders to analyze.")
    args = parser.parse_args()
    main(args.root)

