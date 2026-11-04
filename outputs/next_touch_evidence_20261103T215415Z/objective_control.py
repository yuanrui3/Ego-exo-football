from common import RUN, read, write, prior
from followups import listwise


def run():
    binary=listwise(objective="binary")
    softmax=read(RUN/"adaptive_listwise.json")
    comparison={}
    for target,r in binary.items():
        s=softmax[target]
        per_match={match:{"softmax":s["per_match"][match]["listwise_ensemble"],"binary":r["per_match"][match]["binary_ensemble"]} for match in r["per_match"]}
        simple={"per_match":per_match}
        comparison[target]={"pooled_softmax":s["pooled"]["listwise_ensemble"],"pooled_binary":r["pooled"]["binary_ensemble"],"conditional_NLL":{"softmax":s["conditional_NLL"]["listwise_ensemble"],"binary":r["conditional_NLL"]["binary_ensemble"]},"paired_conditional_AP":prior.contrast(simple,"softmax","binary"),"switch_top1":{"softmax":s["switch_only"]["pooled"]["listwise_ensemble"]["scene"]["top1"],"binary":r["switch_only"]["pooled"]["binary_ensemble"]["scene"]["top1"]},"scope":"Matched architecture/data/seeds/batches/epochs/development criterion; training loss and probability link differ deliberately."}
    write(RUN/"objective_control_comparison.json",comparison)
    return comparison


if __name__=="__main__":
    run()
