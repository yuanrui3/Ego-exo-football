from common import RUN, read, write, digest, prior
from external import external_study, GAME2_FILES, COMMIT
from followups import lock


def run():
    lock("F_frozen_external_replication","E's conditional-choice benefit vanished in unconditional AP, while Game1 physics transfer is stronger than naive proximity. Replicate the fixed source-only decoder on a second previously unscored anonymous match rather than further tuning known matches.","Metrica Game2 geometric1/2s AP, top1 and switch ranking plus separately named PASS actor target; exact Game1 mapping revision2 and no external selection.","Download only61.7MB official pinned Game2 sample with Git blob SHAs. Six frozen all-six-SoccerTrack HGB models reconstructed with identical settings; no Metrica fitting/calibration, no new neural training.")
    path=RUN/"game2_replication.lock.json"
    if not path.exists():
        write(path,{"utc":prior.utc(),"commit":COMMIT,"files":GAME2_FILES,"external_parser_sha256":digest(RUN/"external.py"),"source_features_sha256":digest(RUN/"features_local"/"compact_lag0.npz"),"source_labels_sha256":digest(RUN/"targets_local.npz"),"rule":"Exactly same revision2 mapping as Game1; Game2 not inspected/scored before this lock. No external fitting or threshold/feature changes permitted.","claim":"Frozen second-domain-match replication, not a new SoccerTrack sealed test."})
    frozen=read(path)
    if frozen["external_parser_sha256"]!=digest(RUN/"external.py"):
        raise ValueError("Frozen Game2 parser changed")
    return external_study(game=2)


if __name__=="__main__":
    run()
