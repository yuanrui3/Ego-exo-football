import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


def verify(path, expected=None):
    path=Path(path)
    if expected is None:
        expected=Path(str(path)+".sha256").read_text(encoding="ascii").split()[0]
    h=hashlib.sha256()
    with path.open("rb") as stream:
        while block:=stream.read(8*1024*1024):
            h.update(block)
    actual=h.hexdigest()
    if actual!=expected:
        raise ValueError(f"Archive SHA-256 mismatch: {actual} != {expected}")
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(set(names))!=len(names):
            raise ValueError("Duplicate ZIP member")
        if any(PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts for n in names):
            raise ValueError("Unsafe member path")
        bad=archive.testzip()
        if bad:
            raise ValueError(f"CRC mismatch: {bad}")
        manifests=[n for n in names if n.endswith("/MANIFEST.sha256.json")]
        if len(manifests)!=1:
            raise ValueError("Exactly one member-hash manifest required")
        manifest=json.loads(archive.read(manifests[0]))
        if set(names)!=(set(manifest)|{manifests[0]}):
            raise ValueError("Member set differs from hash manifest")
        for name,expected_member in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=expected_member:
                raise ValueError(f"Member SHA-256 mismatch: {name}")
    return {"status":"PASS","archive":str(path),"sha256":actual,"bytes":path.stat().st_size,"members":len(names),"CRC":"PASS","member_SHA256":"PASS","sidecar_SHA256":"PASS"}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--sha256")
    args=parser.parse_args()
    print(json.dumps(verify(args.archive,args.sha256),indent=2))
