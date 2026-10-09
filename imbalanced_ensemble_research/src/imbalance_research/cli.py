from __future__ import annotations
import argparse, csv, json, math, random, shlex, sys
from dataclasses import asdict
from pathlib import Path

from .core import *


def command_text(): return " ".join(shlex.quote(x) for x in sys.argv)
def markdown_audit(report):
    lines=["# 数据与参考 PDF 只读审计", "", "本报告仅访问配置指定的路径；未知风险不作否定性结论。", ""]
    lines += [f"- 数据目录：`{report['input_boundary'][0]}`（存在：{report['data_dir_exists']}）", f"- PDF：`{report['input_boundary'][1]}`（{report['reference_pdf'].get('status')}）", ""]
    for d in report["data_files"]:
        lines += [f"## {d['file']}", f"- 解析：{d['parse_status']}；尺寸：{d.get('fingerprint',{}).get('bytes','未知')} 字节"]
        if d["parse_status"] == "success": lines += [f"- 维度：{d['dimensions']}；观察到末列：{d['observed_last_column']}；类别：{d['last_column_counts']}", f"- 非数值列：{d['non_numeric_columns']}；缺失：{d['missing_by_column']}；完全重复行：{d['duplicate_rows']}；重复特征行：{d['duplicate_feature_rows']}", f"- 风险：{'；'.join(d['leakage_risks'])}"]
        else: lines.append(f"- 原因：{d['error']}")
        lines.append("")
    lines += ["## PDF 格式观察", f"- {report['reference_pdf']}"]
    return "\n".join(lines)

def make_candidate(rng, D,T,K,g,slot):
    return Candidate(uid=f"g{g}_s{slot}_{rng.randrange(1<<30):08x}",generation=g,slot=slot,f=[rng.random() for _ in range(D)],i=[rng.random() for _ in range(T)],m=[rng.random() for _ in range(K)],w=[rng.random() for _ in range(K)])
def trial_from(target,pop,rng,F,CR,g,slot):
    others=[x for x in pop if x is not target]; a,b,c=rng.sample(others,3)
    raw=a.f+a.i+a.m+a.w; tb=target.f+target.i+target.m+target.w; rb=b.f+b.i+b.m+b.w; rc=c.f+c.i+c.m+c.w
    mandatory=rng.randrange(len(raw)); x=[min(1,max(0, raw[j]+F*(rb[j]-rc[j]))) if (j==mandatory or rng.random()<CR) else tb[j] for j in range(len(raw))]
    D,T,K=len(target.f),len(target.i),len(target.m); return Candidate(uid=f"g{g}_s{slot}_{rng.randrange(1<<30):08x}",generation=g,slot=slot,f=x[:D],i=x[D:D+T],m=x[D+T:D+T+K],w=x[D+T+K:])
def evaluate_pool(candidates,Xtr,ytr,Xv,yv,seed,g,svm_params,complexity_weights,feature_names):
    K=len(candidates); fhash=stable_hash(feature_names); ihash=stable_hash(list(range(len(ytr)))); pool_uid=stable_hash({"generation":g,"candidate_uids":[c.uid for c in candidates],"feature":fhash,"train":ihash})
    specs=[]; scores=[]
    for slot,c in enumerate(candidates):
        _, frep=decode(c.f,"f"); _, irep=decode(c.i,"i",ytr)
        spec=BaseLearnerSpec(pool_uid,g,slot,c.uid,c.f,c.i,(seed * 1009 + g * 100 + slot) % (2**32 - 1),fhash,ihash,svm_params,frep+irep); specs.append(spec)
        try: scores.append(train_and_predict(spec,Xtr,ytr,Xv))
        except Exception as exc:
            scores.append(None)
            spec.repairs.append({"kind":"unavailable","reason":f"{type(exc).__name__}: {exc}"})
    manifest=GenerationPoolManifest(pool_uid,g,tuple(specs),fhash,ihash)
    for c in candidates:
        c.pool_uid=pool_uid; c.manifest=manifest; chosen,_=decode(c.m,"m")
        if any(scores[j] is None for j in chosen): c.valid=False; c.invalid_reason="selected_unavailable_slot"; continue
        raw=[c.w[j] for j in chosen]; total=sum(raw); weights=[z/total for z in raw] if total else [1/len(chosen)]*len(chosen)
        import numpy as np
        prob=np.sum([w*scores[j] for j,w in zip(chosen,weights)],axis=0); met=metric_record(yv,(prob>=.5).astype(int),prob)
        if met["FPR"] is None or met["FNR"] is None: c.valid=False; c.invalid_reason="validation_metric_unavailable"
        else: c.objectives=(met["FPR"],met["FNR"],complexity(c,len(feature_names),len(ytr),K,complexity_weights)); c.valid=True
    return manifest

def prepare_numeric(Xtr,Xv,Xte,feature_names):
    import numpy as np
    try: a=np.asarray(Xtr,dtype=float); b=np.asarray(Xv,dtype=float); c=np.asarray(Xte,dtype=float)
    except ValueError as exc: raise ValueError("存在未配置的非数值字段；当前运行需在 preprocessing.categorical_columns 实现训练集编码: "+str(exc))
    finite=np.isfinite(a); med=np.nanmedian(np.where(finite,a,np.nan),axis=0)
    if not np.all(np.isfinite(med)): raise ValueError("训练集中存在全缺失特征")
    def fix(z): return np.where(np.isfinite(z),z,med)
    a,b,c=fix(a),fix(b),fix(c); keep=np.nanstd(a,axis=0)>0
    if not keep.any(): raise ValueError("训练预处理后没有非常量特征")
    mean,std=a[:,keep].mean(0),a[:,keep].std(0); return (a[:,keep]-mean)/std,(b[:,keep]-mean)/std,(c[:,keep]-mean)/std,[n for n,k in zip(feature_names,keep) if k],{"fit_scope":"train_only","dropped_constant":[n for n,k in zip(feature_names,keep) if not k]}
def run_baselines(Xtr,ytr,Xte,yte,seed,cfg):
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    models=[("rbf_svm",SVC(kernel="rbf",random_state=seed,**cfg["models"]["rbf_svm"])),("weighted_rbf_svm",SVC(kernel="rbf",random_state=seed,class_weight="balanced",**{k:v for k,v in cfg["models"]["rbf_svm"].items() if k!="class_weight"})),("random_forest",RandomForestClassifier(random_state=seed,**cfg["models"]["random_forest"]))]
    rows=[]
    for name,model in models:
        try: model.fit(Xtr,ytr); p=model.predict(Xte); score=model.predict_proba(Xte)[:,1] if hasattr(model,"predict_proba") else None; rows.append({"model":name,"status":"success",**metric_record(yte,p,score)})
        except Exception as exc: rows.append({"model":name,"status":"not_run_runtime_error","reason":f"{type(exc).__name__}: {exc}"})
    try:
        from imblearn.ensemble import RUSBoostClassifier
        model=RUSBoostClassifier(random_state=seed,**cfg["models"]["rusboost"]); model.fit(Xtr,ytr); score=model.predict_proba(Xte)[:,1]; rows.append({"model":"rusboost","status":"success",**metric_record(yte,model.predict(Xte),score)})
    except ImportError: rows.append({"model":"rusboost","status":"not_run_dependency_missing","reason":"需 imbalanced-learn==0.12.4"})
    except Exception as exc: rows.append({"model":"rusboost","status":"not_run_runtime_error","reason":f"{type(exc).__name__}: {exc}"})
    return rows
def run_one(spec,cfg,seed):
    from sklearn.model_selection import train_test_split
    X,y,features,meta=load_dataset(spec)
    if min(y.count(0),y.count(1))<10: raise ValueError("每类至少需要10条记录以支持两次分层拆分")
    indices=list(range(len(y))); tr,temp=train_test_split(indices,test_size=.3,stratify=y,random_state=seed); va,te=train_test_split(temp,test_size=1/3,stratify=[y[i] for i in temp],random_state=seed)
    Xtr0,Xv0,Xte0=([X[i] for i in z] for z in (tr,va,te)); ytr,yv,yte=([y[i] for i in z] for z in (tr,va,te))
    Xtr,Xv,Xte,features,pre=prepare_numeric(Xtr0,Xv0,Xte0,features); base=run_baselines(Xtr,ytr,Xte,yte,seed,cfg)
    result={"dataset":spec["id"],"seed":seed,"input":fingerprint(spec["path"]),"label_mapping":meta["label_mapping"],"split_summary":{"requested":[.7,.2,.1],"train":len(tr),"validation":len(va),"test":len(te),"train_class_counts":{str(k):ytr.count(k) for k in (0,1)},"validation_class_counts":{str(k):yv.count(k) for k in (0,1)},"test_class_counts":{str(k):yte.count(k) for k in (0,1)},"indices":{"train":tr,"validation":va,"test":te}},"preprocessing":pre,"baselines":base,"optimizer":None}
    if not cfg["optimization"].get("enabled",True): return result
    o=cfg["optimization"]; rng=random.Random(seed); pop=[make_candidate(rng,len(features),len(ytr),o["K"],0,j) for j in range(o["K"])]
    initial_manifest=evaluate_pool(pop,Xtr,ytr,Xv,yv,seed,0,cfg["models"]["rbf_svm"],o["complexity_weights"],features)
    for g in range(1,o["generations"]+1):
        trials=[trial_from(t,pop,rng,o["F"],o["CR"],g,j) for j,t in enumerate(pop)]; evaluate_pool(trials,Xtr,ytr,Xv,yv,seed,g,cfg["models"]["rbf_svm"],o["complexity_weights"],features)
        survivors=[]
        for target,trial in zip(pop,trials):
            if trial.valid and (not target.valid or all(a<=b for a,b in zip(trial.objectives,target.objectives)) and any(a<b for a,b in zip(trial.objectives,target.objectives))): survivors.append(trial)
            elif target.valid and trial.valid and not (all(a<=b for a,b in zip(target.objectives,trial.objectives)) and any(a<b for a,b in zip(target.objectives,trial.objectives))): survivors.append(min((target,trial),key=lambda c:(sum(c.objectives),c.uid)))
            else: survivors.append(target)
        pop=survivors
    front=nondominated(pop); decision=promethee(front,cfg["promethee"]); opt_report={"algorithm":"spMODE-II-inspired","initialization_pool_count":1,"evolution_generation_count":o["generations"],"pools_per_evolution_generation":1,"pool_manifest_initial":asdict(initial_manifest),"candidates":[asdict(c) for c in pop],"promethee":decision}
    if decision["status"] != "no_candidate":
        selected=next(c for c in front if c.uid==decision["selected_uid"]); scores,final=retrain_selected_candidate(selected,selected.manifest,Xtr,ytr,Xte); opt_report["selected_candidate"]={"candidate":asdict(selected),"pool_manifest":asdict(selected.manifest),"final_retraining":final,"test_metrics":metric_record(yte,(scores>=.5).astype(int),scores)}
    result["optimizer"]=opt_report; return result

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("mode",choices=["audit","check-env","run","review"]); parser.add_argument("--config",required=True); args=parser.parse_args(argv)
    try: cfg,_=load_config(args.config)
    except Exception as exc: print("配置错误:",exc,file=sys.stderr); return 2
    art=Artifacts(cfg,command_text(),args.mode)
    try:
        art.write_json("environment.json",environment_report())
        if args.mode=="check-env": art.write_json("environment.report.json",environment_report()); art.write_text("environment.md","# 依赖状态报告\n\n```json\n"+json.dumps(environment_report(),ensure_ascii=False,indent=2)+"\n```\n"); art.status("success","environment"); print(art.path); return 0
        if args.mode=="audit":
            report=audit_inputs(cfg); art.write_json("audit.json",report); art.write_text("audit.md",markdown_audit(report)); art.status("success","audit",input_fingerprint={"data_dir":fingerprint(cfg['inputs']['data_dir']),"reference_pdf":fingerprint(cfg['inputs']['reference_pdf'])}); print(art.path); return 0
        if args.mode=="review": art.write_text("review.md",Path("docs/代码审查记录.md").read_text(encoding="utf-8")); art.status("success","review"); print(art.path); return 0
        if not cfg["datasets"]: raise ValueError("run 要求 datasets 至少包含一个已确认标签列与正类映射的数据集")
        all_results=[]
        for spec in cfg["datasets"]:
            for seed in cfg["experiment"]["seeds"]: all_results.append(run_one(spec,cfg,seed))
        art.write_json("results.json",all_results); art.write_json("input_fingerprint.json",{d["id"]:fingerprint(d["path"]) for d in cfg["datasets"]})
        rows=[]
        for r in all_results:
            for b in r["baselines"]: rows.append({"dataset":r["dataset"],"seed":r["seed"],**b})
        art.write_text("metrics.csv","\n".join([",".join(rows[0].keys())]+[",".join(json.dumps(x.get(k,""),ensure_ascii=False) for k in rows[0].keys()) for x in rows]) if rows else "")
        art.status("success","run",datasets=[r["dataset"] for r in all_results]); print(art.path); return 0
    except Exception as exc:
        art.write_text("error.traceback.txt",traceback.format_exc()); art.status("failed",args.mode,error_type=type(exc).__name__,message=str(exc),recoverable=True); print(f"运行失败，详情: {art.path}",file=sys.stderr); return 5
if __name__ == "__main__": raise SystemExit(main())
