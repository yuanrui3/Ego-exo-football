import argparse
import json
import subprocess
import sys
from pathlib import Path

RUN=Path(__file__).resolve().parent


def run(script,*args):
    subprocess.run([sys.executable,str(RUN/script),*args],check=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--all",action="store_true");args=parser.parse_args()
    from common import ROOT,read,digest
    hashes=read(RUN/"source_hashes_before.json")
    for path,sha in hashes.items():
        if digest(ROOT/path)!=sha:raise ValueError("Protected source changed: "+path)
    if not args.all:
        print(json.dumps({"protected_hashes":"PASS","files":len(hashes),"stages":read(RUN/"stage_status.json")},indent=2))
        return
    for stage in range(1,9):run("run.py","--stage",str(stage))
    for script in ("longer.py","semantics.py","context_interpretation.py","historical_team_audit.py","external_contact_audit.py"):
        run(script)
    run("run.py","--stage","9")
    for script in ("domain_resolution.py","conditional_factorization.py","external_replication.py","objective_control.py","negative_controls.py"):
        run(script)
    subprocess.run([sys.executable,"-m","pytest",str(RUN/"test_evidence.py"),"-q","--junitxml="+str(RUN/"tests.xml")],check=True)
    run("figures.py")
    run("finalize.py","--package")


if __name__=="__main__":
    main()
