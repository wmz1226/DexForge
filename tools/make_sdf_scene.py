#!/usr/bin/env python3
"""Write ``<object>_sdf.xml``: a physics scene whose object collides through its triangle mesh.

The scene is the Gaussian scene ``<object>_gs.xml`` with only the object's colliding ``gs`` geom
replaced by an ``sdfmesh`` geom on the object's OBJ (by default the mesh asset of the object's
visual geom). Hand, physics parameters and body ordering are unchanged, so ContactAware and
ForceAware select between the two targets by scene name only.
"""

import argparse
from pathlib import Path
import re
import xml.etree.ElementTree as ET

OBJECT_BODY = "obj"


def make_sdf_scene(gs_xml: Path, obj_file: str | None = None) -> Path:
    gs_xml = Path(gs_xml).resolve()
    if not gs_xml.name.endswith("_gs.xml"):
        raise ValueError(f"Expected an <object>_gs.xml scene, got {gs_xml}")
    text = gs_xml.read_text()
    root = ET.fromstring(text)
    body = root.find(f".//body[@name='{OBJECT_BODY}']")
    if body is None:
        raise ValueError(f"No body named {OBJECT_BODY!r} in {gs_xml}")
    colliding = [geom for geom in body.findall("geom") if geom.get("type") == "gs"]
    if len(colliding) != 1:
        raise ValueError(f"Expected one gs geom on the object in {gs_xml}")
    if obj_file is None:
        assets = {mesh.get("name"): mesh.get("file") for mesh in root.iter("mesh") if mesh.get("file")}
        visual = [assets.get(geom.get("mesh")) for geom in body.findall("geom") if geom.get("type") == "mesh"]
        visual = [path for path in visual if path and path.lower().endswith(".obj")]
        if len(visual) != 1:
            raise ValueError(f"Pass --obj: the object has no unique OBJ mesh asset in {gs_xml}")
        obj_file = visual[0]
    name = colliding[0].get("name")
    match = re.search(rf'<geom\b[^>]*\bname="{re.escape(name)}"[^>]*/>', text)
    replacement = re.sub(r'\btype="gs"', 'type="sdfmesh"', match.group(0))
    replacement = re.sub(r'\bfile="[^"]*"', f'file="{obj_file}"', replacement)
    replacement = re.sub(r'\s+size="[^"]*"', "", replacement)
    target = gs_xml.with_name(gs_xml.name.replace("_gs.xml", "_sdf.xml"))
    target.write_text(text[:match.start()] + replacement + text[match.end():])
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gs_xml", type=Path)
    parser.add_argument("--obj", help="OBJ file, relative to the scene XML")
    args = parser.parse_args()
    print(make_sdf_scene(args.gs_xml, args.obj))


if __name__ == "__main__":
    main()
