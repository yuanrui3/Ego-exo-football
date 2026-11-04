import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RUN, FIT, prior, source, read, write, groups
from experiments import compact_features


def save(name):
    plt.tight_layout()
    plt.savefig(RUN/"figures"/(name+".png"),dpi=140)
    plt.close()


def run():
    (RUN/"figures").mkdir(exist_ok=True)
    qa=read(RUN/"clock_and_sync_audit.json")["halves"]
    keys=list(qa)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].bar(np.arange(len(keys)),[qa[k]["actor_repeated_position_fraction"] for k in keys])
    ax[1].bar(np.arange(len(keys)),[qa[k]["ball_missing_fraction"] for k in keys])
    ax[0].set(title="Repeated native player positions",ylabel="Fraction of adjacent observations")
    ax[1].set(title="Ball unavailable / outside bounds",ylabel="Fraction of native frames")
    for a in ax:a.set_xlabel("Anonymous half index")
    save("01_native_quality")
    plt.figure(figsize=(7,4))
    offsets=(-5,-1,0,1,5)
    for i,k in enumerate(keys):
        plt.plot(np.array(offsets)/25,[qa[k]["offset_diagnostics"][str(o)]["actor_ball_distance_m"].get("p50",np.nan) for o in offsets],alpha=.5)
    plt.xlabel("Diagnostic joint actor/ball shift (s)");plt.ylabel("Median event-actor distance (m)")
    plt.title("Distance is NOT independent synchronization evidence")
    save("02_clock_diagnostic")
    contact=read(RUN/"contact_candidate_audit.json")["halves"]
    ext=read(RUN/"external_contact_association.json")
    names=("proximity","intercept","changepoint")
    native=np.array([[v["candidate_frames"][n]/v["valid_ball_frames"] for n in names] for v in contact.values()])
    plt.figure(figsize=(7,4));i=np.arange(3)
    plt.bar(i-.18,native.mean(0),.36,label="SoccerTrack mean across halves")
    plt.bar(i+.18,[ext["fraction_of_valid_frames"][n] for n in names],.36,label="Metrica one match")
    plt.xticks(i,names);plt.ylabel("Candidate-state fraction of valid frames");plt.legend()
    plt.title("Candidate states are not distinct physical touches")
    save("03_contact_candidate_states")
    targets=read(RUN/"next_interactor_results.json")
    long=read(RUN/"longer_horizon_results.json")
    plt.figure(figsize=(7,4))
    for model in ("nearest","compact_current","interpretable"):
        vals=[targets["geometry_1s"],targets["geometry_2s"],long["geometry_3s"],long["geometry_5s"]]
        plt.plot([1,2,3,5],[v["macro_match"][model]["AP"] for v in vals],marker="o",label=model)
    plt.xlabel("Geometric forecast horizon (s)");plt.ylabel("Six-match macro AP")
    plt.title("Future nearest-player proxy, not a touch label");plt.legend()
    save("04_geometric_lead_curve")
    action=read(RUN/"anticipation_challenges.json")
    plt.figure(figsize=(7,4))
    for model in ("nearest","current","raw_history","interpretable","compact_kinematic","joint_stale_0p48s"):
        plt.plot([.5,1.5,2.5],[action[h]["macro_match"][model]["AP"] for h in ("lead_0_1s","lead_1_2s","lead_2_3s")],marker="o",label=model)
    plt.xlabel("Midpoint of disjoint future action window (s)");plt.ylabel("Six-match macro AP");plt.legend(fontsize=8)
    plt.title("Supported BAS any-action anticipation")
    save("05_action_anticipation")
    context=read(RUN/"tactical_context_results.json")
    fig,ax=plt.subplots(1,3,figsize=(13,4))
    for a,(target,r) in zip(ax,context.items()):
        models=list(r["macro_match"])
        a.bar(range(len(models)),[r["macro_match"][m]["AP"] for m in models])
        title={"early_action":"1-2 s any BAS action","geometric_switch_risk":"Future nearest at 2 s (all)","next_BAS_actor":"1-2 s next BAS actor"}[target]
        a.set_xticks(range(len(models)),models,rotation=35,ha="right");a.set_title(title);a.set_ylabel("Six-match macro AP")
    save("06_context_ablation")
    scene=read(RUN/"scene_prioritization.json")
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for a,(h,r) in zip(ax,scene.items()):
        for model,v in r["workload"].items():
            a.plot([5,10,20],[v[f"{b}_top3"]["event_recall"] for b in (5,10,20)],marker="o",label=model)
        a.set(title=h,xlabel="Scenes / match hour",ylabel="Distinct annotated-event recall");a.legend(fontsize=8)
    save("07_review_workload")
    external=read(RUN/"external_results.json")
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for a,target in zip(ax,("geometry_1s","geometry_2s")):
        r=external["results"][target]["models"];models=list(r)
        a.bar(range(len(models)),[r[m]["all"]["AP"] for m in models]);a.set_xticks(range(len(models)),models,rotation=30,ha="right")
        a.set(title=target,ylabel="External one-match AP")
    save("08_external_pilot")
    semantics=read(RUN/"conditional_action_type.json")
    plt.figure(figsize=(7,4))
    for model in ("train_class_prior","kinematic","full_context"):
        plt.plot([.5,1.5,2.5],[semantics[h]["macro_match"][model]["macro_F1"] for h in semantics],marker="o",label=model)
    plt.legend();plt.xlabel("Future window midpoint (s)");plt.ylabel("Six-match macro F1")
    plt.title("Action type GIVEN annotated occurrence")
    save("09_conditional_action_type")
    plt.figure(figsize=(9,4))
    support=action["lead_1_2s"]["raw_class_support"]
    plt.bar(range(len(support)),list(support.values()))
    plt.xticks(range(len(support)),list(support),rotation=25,ha="right")
    plt.ylabel("Positive actor-row labels");plt.title("Raw BAS class support, 1-2 s window")
    save("10_class_support")
    d=prior.data();f=compact_features(d)
    with np.load(RUN/"predictions_local"/"lead_1_2s.npz") as z:p=z["interpretable"]
    rng=np.random.default_rng(20261103);valid=np.flatnonzero(np.isfinite(p)&(f["state"][:,0]>.5))
    take=rng.choice(valid,min(6000,len(valid)),replace=False)
    plt.figure(figsize=(7,4));plt.hexbin(f["state"][take,4]*105,p[take],gridsize=45,mincnt=1,bins="log")
    plt.colorbar(label="log count");plt.xlim(0,40);plt.xlabel("Current actor-ball distance (m)");plt.ylabel("OOF upcoming-action probability");plt.title("Scores are not simply the nearest player")
    save("11_score_vs_distance")
    if (RUN/"adaptive_calibration.json").exists():
        cal=read(RUN/"adaptive_calibration.json")
        plt.figure(figsize=(6,4));plt.plot([0,1],[0,1],"k--",alpha=.4)
        for model,v in cal["reliability"].items():
            plt.plot([b["predicted"] for b in v["bins"]],[b["observed"] for b in v["bins"]],marker="o",label=model)
        plt.legend();plt.xlabel("Mean prediction");plt.ylabel("Observed annotation rate");plt.title("Annotation-relative reliability, not contact probability")
        save("12_calibration")
    if (RUN/"adaptive_listwise.json").exists():
        nn=read(RUN/"adaptive_listwise.json")
        fig,ax=plt.subplots(1,2,figsize=(10,4))
        for a,(target,r) in zip(ax,nn.items()):
            models=("hgb","listwise_ensemble")
            a.bar([0,1],[r["switch_only"]["pooled"][m]["scene"]["top1"] for m in models])
            a.set_xticks([0,1],models);a.set(title=target,ylabel="Conditional switch top1")
        save("13_objective_alignment")
    importance=read(RUN/"context_interpretation.json")["mean_group_AP_drop"]
    plt.figure(figsize=(9,4));plt.bar(range(len(importance)),list(importance.values()))
    plt.xticks(range(len(importance)),list(importance),rotation=25,ha="right")
    plt.ylabel("Held-match AP drop after group permutation")
    plt.title("Correlated-group sensitivity, not causal importance")
    save("14_context_sensitivity")
    if (RUN/"objective_control_comparison.json").exists():
        control=read(RUN/"objective_control_comparison.json")
        fig,ax=plt.subplots(1,2,figsize=(10,4))
        for a,(target,r) in zip(ax,control.items()):
            a.bar([0,1],[r["switch_top1"]["binary"],r["switch_top1"]["softmax"]])
            a.set_xticks([0,1],["Binary BCE","Scene softmax"]);a.set(title=target,ylabel="Conditional switch top1")
        save("15_matched_objective_control")
    if (RUN/"external_results_game2.json").exists():
        second=read(RUN/"external_results_game2.json")
        fig,ax=plt.subplots(1,2,figsize=(10,4))
        for a,target in zip(ax,("geometry_1s","geometry_2s")):
            for i,game in enumerate((external,second)):
                a.bar(np.arange(2)+i*.35,[game["results"][target]["models"][m]["all"]["AP"] for m in ("nearest","kinematic")],width=.35,label=f"Game{i+1}")
            a.set_xticks(np.arange(2)+.175,["Nearest","Frozen kinematic HGB"]);a.set(title=target,ylabel="External AP");a.legend()
        save("16_frozen_external_replication")
    if (RUN/"adaptive_unconditional_factorization.json").exists():
        fact=read(RUN/"adaptive_unconditional_factorization.json")
        models=list(fact["macro_match"])
        plt.figure(figsize=(8,4));plt.bar(range(len(models)),[fact["macro_match"][m]["AP"] for m in models])
        plt.xticks(range(len(models)),["Original HGB","Scene-calibrated HGB","Same scene + softmax allocation"],rotation=12)
        plt.ylabel("Six-match macro AP");plt.title("Conditional gains do not imply unconditional gains")
        save("17_unconditional_negative_result")
    manifest=[];case_index=0
    for match in FIT:
        for halfcode,half in ((1,"1st"),(2,"2nd")):
            if case_index>=8:break
            mask=(d["match"]==match)&(d["half_code"]==halfcode)
            qs=np.unique(d["query_frame"][mask]);q=int(qs[len(qs)//2])
            rows=np.flatnonzero(mask&(d["query_frame"]==q))
            raw,track=source(match,half)
            active=np.flatnonzero(raw["observed"][q])
            b,_,_,ok=track.asof(np.array([q]))
            plt.figure(figsize=(9,6));a=plt.gca();a.set_facecolor("#e8f4e3")
            a.plot([-52.5,52.5,52.5,-52.5,-52.5],[-34,-34,34,34,-34],"k-",alpha=.5)
            a.plot([0,0],[-34,34],"k-",alpha=.3)
            colors=np.where(raw["teams"][q,active]==1,"#247ba0","#ed553b")
            a.scatter(raw["xy"][q,active,0],raw["xy"][q,active,1],c=colors,s=55)
            if ok[0]:a.scatter(*b[0],c="black",s=65,marker="x")
            times=np.arange(max(0,q-50),min(len(raw["xy"]),q+51))
            positions,_,_,good=track.asof(times)
            for past,style in ((True,"-"),(False,"--")):
                use=good&((times<=q) if past else (times>q))
                a.plot(positions[use,0],positions[use,1],style,color="black",alpha=.6,label="past input" if past else "future reference only")
            top=rows[np.argsort(-p[rows],kind="stable")[:3]]
            for index,row in enumerate(top):
                slot=int(d["actor_slot_local_only"][row]);pos=raw["xy"][q,slot]
                a.annotate(f"rank{index+1}: {p[row]:.3f}",pos,xytext=(4,6),textcoords="offset points",fontsize=8)
            case_index+=1
            a.set(xlim=(-55,55),ylim=(-37,37),aspect="equal",title=f"Anonymous C{case_index:02d}: source query frame {q}, half {halfcode}")
            a.legend(loc="lower right");a.set_xlabel("Pitch metres. Solid history ends at q; dashed future is evaluation reference only.\nOOF 1-2 s BAS scores; quantized source positions; physical synchronization unverified.")
            save(f"case_C{case_index:02d}")
            manifest.append({"case":f"C{case_index:02d}","match":match,"half":half,"query_frame":q,"selection":"middle eligible scene of each ordered half, independent of labels/model outcomes"})
    write(RUN/"case_manifest_local.json",manifest)
    result={"diagnostics":len(list((RUN/"figures").glob("[0-9][0-9]_*.png"))),"anonymous_case_diagrams":len(manifest)}
    write(RUN/"figure_inventory.json",result)
    return result


if __name__=="__main__":
    print(run())
