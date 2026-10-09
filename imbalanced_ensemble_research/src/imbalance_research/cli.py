from __future__ import annotations
import argparse, json, random, shlex, sys
from dataclasses import asdict
from pathlib import Path

from .core import *


MISSING_VALUES = {"", "?", "NA", "NaN"}


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


def make_candidate(rng, D, T, K, g, slot):
    return Candidate(uid=f"g{g}_s{slot}_{rng.randrange(1<<30):08x}", generation=g, slot=slot, f=[rng.random() for _ in range(D)], i=[rng.random() for _ in range(T)], m=[rng.random() for _ in range(K)], w=[rng.random() for _ in range(K)])


def trial_from(target, pop, rng, F, CR, g, slot):
    others=[x for x in pop if x is not target]; a,b,c=rng.sample(others,3)
    raw=a.f+a.i+a.m+a.w; tb=target.f+target.i+target.m+target.w; rb=b.f+b.i+b.m+b.w; rc=c.f+c.i+c.m+c.w
    mandatory=rng.randrange(len(raw)); x=[min(1,max(0, raw[j]+F*(rb[j]-rc[j]))) if (j==mandatory or rng.random()<CR) else tb[j] for j in range(len(raw))]
    D,T,K=len(target.f),len(target.i),len(target.m)
    return Candidate(uid=f"g{g}_s{slot}_{rng.randrange(1<<30):08x}", generation=g, slot=slot, f=x[:D], i=x[D:D+T], m=x[D+T:D+T+K], w=x[D+T+K:])


def split_indices(y, seed):
    """Create the required mutually exclusive raw-record 7:2:1 stratified split."""
    by_class={}
    for index, label in enumerate(y): by_class.setdefault(label,[]).append(index)
    train=[]; validation=[]; test=[]; rng=random.Random(seed)
    for label, indices in sorted(by_class.items()):
        shuffled=list(indices); rng.shuffle(shuffled)
        train_count=round(len(shuffled)*.7); temporary=shuffled[train_count:]
        validation_count=round(len(temporary)*2/3)
        train.extend(shuffled[:train_count]); validation.extend(temporary[:validation_count]); test.extend(temporary[validation_count:])
    rng.shuffle(train); rng.shuffle(validation); rng.shuffle(test)
    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test) or set(train) | set(validation) | set(test) != set(range(len(y))):
        raise RuntimeError("分层拆分未形成互斥完整分区")
    return train, validation, test


def evaluate_pool(candidates, Xtr, ytr, Xv, yv, seed, g, svm_params, complexity_weights, feature_names, threshold):
    """Evaluate candidates with train/validation only; this interface has no test input."""
    K=len(candidates); fhash=stable_hash(feature_names); ihash=stable_hash(list(range(len(ytr))))
    pool_uid=stable_hash({"generation":g,"candidate_uids":[c.uid for c in candidates],"feature":fhash,"train":ihash})
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
        prob=np.sum([w*np.asarray(scores[j], dtype=float) for j,w in zip(chosen,weights)],axis=0)
        met=metric_record(yv,(prob>=threshold).astype(int),prob,threshold)
        if met["FPR"] is None or met["FNR"] is None: c.valid=False; c.invalid_reason="validation_metric_unavailable"
        else: c.objectives=(met["FPR"],met["FNR"],complexity(c,len(feature_names),len(ytr),K,complexity_weights)); c.valid=True
    return manifest


def _numeric_values(values, column):
    import numpy as np
    parsed=[]
    for value in values:
        if value in MISSING_VALUES: parsed.append(np.nan); continue
        try: parsed.append(float(value))
        except (TypeError, ValueError): raise ValueError(f"数值列 {column} 包含无法转换的值: {value!r}")
    return np.asarray(parsed, dtype=float)


def _categorical_values(values):
    return ["__MISSING__" if value in MISSING_VALUES else str(value) for value in values]


def prepare_numeric(Xtr, Xv, Xte, feature_names, preprocessing):
    """Fit deletion, imputation, one-hot encoding and scaling on training rows only."""
    import numpy as np
    policy=validate_preprocessing_columns(feature_names, preprocessing)
    groups=policy["configured"]
    positions={name: index for index,name in enumerate(feature_names)}
    pieces_train=[]; pieces_validation=[]; pieces_test=[]; encoded_names=[]
    numeric_imputation={}; categorical_encoding={}; dropped=[]
    for column in groups["numeric_columns"]:
        index=positions[column]; train=_numeric_values([row[index] for row in Xtr],column)
        validation=_numeric_values([row[index] for row in Xv],column); test=_numeric_values([row[index] for row in Xte],column)
        finite=train[np.isfinite(train)]
        if not len(finite):
            dropped.append({"column":column,"reason":"train_all_missing"}); continue
        median=float(np.median(finite)); numeric_imputation[column]={"strategy":"train_median","value":median}
        pieces_train.append(np.where(np.isfinite(train),train,median).reshape(-1,1))
        pieces_validation.append(np.where(np.isfinite(validation),validation,median).reshape(-1,1))
        pieces_test.append(np.where(np.isfinite(test),test,median).reshape(-1,1)); encoded_names.append(column)
    for column in groups["categorical_columns"]:
        index=positions[column]; train=_categorical_values([row[index] for row in Xtr])
        validation=_categorical_values([row[index] for row in Xv]); test=_categorical_values([row[index] for row in Xte])
        observed=sorted(set(value for value in train if value != "__MISSING__"))
        if not observed:
            dropped.append({"column":column,"reason":"train_all_missing"}); continue
        categories=sorted(set(train)); name_map=[f"{column}={value}" for value in categories]
        categorical_encoding[column]={"strategy":"train_categories_one_hot","missing_value":"__MISSING__","categories":categories,"unknown_handling":"all_zero","validation_unknown_count":sum(value not in categories for value in validation),"test_unknown_count":sum(value not in categories for value in test)}
        def encode(values): return np.asarray([[1.0 if value == category else 0.0 for category in categories] if value in categories else [0.0]*len(categories) for value in values])
        pieces_train.append(encode(train)); pieces_validation.append(encode(validation)); pieces_test.append(encode(test)); encoded_names.extend(name_map)
    if not pieces_train: raise ValueError("按配置删除/空列处理后没有可用特征")
    train=np.concatenate(pieces_train,axis=1); validation=np.concatenate(pieces_validation,axis=1); test=np.concatenate(pieces_test,axis=1)
    keep=np.std(train,axis=0)>0
    for name, retained in zip(encoded_names,keep):
        if not retained: dropped.append({"column":name,"reason":"train_constant_after_imputation_or_encoding"})
    if not keep.any(): raise ValueError("训练预处理后没有非常量特征")
    mean=train[:,keep].mean(0); std=train[:,keep].std(0)
    audit={
        "fit_scope":"train_only",
        "raw_column_policy":policy,
        "dropped_by_configuration":{"drop_columns":groups["drop_columns"],"identifier_columns":groups["identifier_columns"]},
        "numeric_imputation":numeric_imputation,
        "categorical_encoding":categorical_encoding,
        "dropped_columns":dropped,
        "output_features":[name for name, retained in zip(encoded_names,keep) if retained],
        "standardization":{"fit_scope":"train_only","mean":mean.tolist(),"std":std.tolist()}
    }
    return (train[:,keep]-mean)/std, (validation[:,keep]-mean)/std, (test[:,keep]-mean)/std, audit["output_features"], audit


def run_baselines(Xtr, ytr, Xte, yte, seed, cfg, threshold):
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    models=[("rbf_svm",SVC(kernel="rbf",random_state=seed,**cfg["models"]["rbf_svm"])),("weighted_rbf_svm",SVC(kernel="rbf",random_state=seed,class_weight="balanced",**{k:v for k,v in cfg["models"]["rbf_svm"].items() if k!="class_weight"})),("random_forest",RandomForestClassifier(random_state=seed,**cfg["models"]["random_forest"]))]
    rows=[]
    for name,model in models:
        try:
            model.fit(Xtr,ytr); score=model.predict_proba(Xte)[:,list(model.classes_).index(1)]
            rows.append({"model":name,"status":"success",**metric_record(yte,(score>=threshold).astype(int),score,threshold)})
        except Exception as exc: rows.append({"model":name,"status":"not_run_runtime_error","reason":f"{type(exc).__name__}: {exc}","decision_threshold":threshold})
    try:
        from imblearn.ensemble import RUSBoostClassifier
        model=RUSBoostClassifier(random_state=seed,**cfg["models"]["rusboost"]); model.fit(Xtr,ytr); score=model.predict_proba(Xte)[:,list(model.classes_).index(1)]
        rows.append({"model":"rusboost","status":"success",**metric_record(yte,(score>=threshold).astype(int),score,threshold)})
    except ImportError: rows.append({"model":"rusboost","status":"not_run_dependency_missing","reason":"需 imbalanced-learn==0.12.4","decision_threshold":threshold})
    except Exception as exc: rows.append({"model":"rusboost","status":"not_run_runtime_error","reason":f"{type(exc).__name__}: {exc}","decision_threshold":threshold})
    return rows


def run_one(spec, cfg, seed):
    X,y,features,meta=load_dataset(spec)
    if min(y.count(0),y.count(1))<10: raise ValueError("每类至少需要10条记录以支持两次分层拆分")
    train_indices,validation_indices,test_indices=split_indices(y,seed)
    Xtr0,Xv0,Xte0=([X[i] for i in indices] for indices in (train_indices,validation_indices,test_indices))
    ytr,yv,yte=([y[i] for i in indices] for indices in (train_indices,validation_indices,test_indices))
    Xtr,Xv,Xte,features,pre=prepare_numeric(Xtr0,Xv0,Xte0,features,cfg["preprocessing"])
    threshold=cfg["optimization"]["threshold"]
    base=run_baselines(Xtr,ytr,Xte,yte,seed,cfg,threshold)
    result={"dataset":spec["id"],"seed":seed,"input":fingerprint(spec["path"]),"label_mapping":meta["label_mapping"],"decision_threshold":threshold,"split_summary":{"requested":[.7,.2,.1],"train":len(train_indices),"validation":len(validation_indices),"test":len(test_indices),"mutually_exclusive":True,"train_class_counts":{str(k):ytr.count(k) for k in (0,1)},"validation_class_counts":{str(k):yv.count(k) for k in (0,1)},"test_class_counts":{str(k):yte.count(k) for k in (0,1)},"indices":{"train":train_indices,"validation":validation_indices,"test":test_indices}},"preprocessing":pre,"baselines":base,"optimizer":None}
    if not cfg["optimization"].get("enabled",True): return result
    o=cfg["optimization"]; rng=random.Random(seed); pop=[make_candidate(rng,len(features),len(ytr),o["K"],0,j) for j in range(o["K"])]
    manifests=[]
    initial_manifest=evaluate_pool(pop,Xtr,ytr,Xv,yv,seed,0,cfg["models"]["rbf_svm"],o["complexity_weights"],features,threshold); manifests.append(initial_manifest)
    for g in range(1,o["generations"]+1):
        trials=[trial_from(target,pop,rng,o["F"],o["CR"],g,j) for j,target in enumerate(pop)]
        manifests.append(evaluate_pool(trials,Xtr,ytr,Xv,yv,seed,g,cfg["models"]["rbf_svm"],o["complexity_weights"],features,threshold))
        survivors=[]
        for target,trial in zip(pop,trials):
            if trial.valid and (not target.valid or all(a<=b for a,b in zip(trial.objectives,target.objectives)) and any(a<b for a,b in zip(trial.objectives,target.objectives))): survivors.append(trial)
            elif target.valid and trial.valid and not (all(a<=b for a,b in zip(target.objectives,trial.objectives)) and any(a<b for a,b in zip(target.objectives,trial.objectives))): survivors.append(min((target,trial),key=lambda c:(sum(c.objectives),c.uid)))
            else: survivors.append(target)
        pop=survivors
    front=nondominated(pop); decision=promethee(front,cfg["promethee"])
    opt_report={"algorithm":"spMODE-II-inspired","decision_threshold":threshold,"evaluation_data_scope":{"training":"base learner fit only","validation":"candidate objectives and PROMETHEE input","test":"not passed to evaluate_pool or PROMETHEE"},"initialization_pool_count":1,"evolution_generation_count":o["generations"],"pools_per_evolution_generation":1,"pool_count":len(manifests),"pool_manifests":[asdict(manifest) for manifest in manifests],"candidates":[asdict(candidate) for candidate in pop],"promethee":decision}
    if decision["status"] != "no_candidate":
        selected=next(candidate for candidate in front if candidate.uid==decision["selected_uid"])
        scores,final=retrain_selected_candidate(selected,selected.manifest,Xtr,ytr,Xte)
        opt_report["selected_candidate"]={"candidate":asdict(selected),"pool_manifest":asdict(selected.manifest),"final_retraining":final,"test_metrics":metric_record(yte,(scores>=threshold).astype(int),scores,threshold)}
    result["optimizer"]=opt_report
    return result


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
        if cfg.get("requires_user_label_semantics_confirmation"):
            raise ValueError("formal.template 配置待用户确认全部数据集的标签业务语义，CLI 拒绝运行")
        if not cfg["datasets"]: raise ValueError("run 要求 datasets 至少包含一个已确认标签列与正类映射的数据集")
        all_results=[]
        for spec in cfg["datasets"]:
            for seed in cfg["experiment"]["seeds"]: all_results.append(run_one(spec,cfg,seed))
        art.write_json("results.json",all_results); art.write_json("input_fingerprint.json",{dataset["id"]:fingerprint(dataset["path"]) for dataset in cfg["datasets"]})
        rows=[]
        for result in all_results:
            for baseline in result["baselines"]: rows.append({"dataset":result["dataset"],"seed":result["seed"],**baseline})
        art.write_text("metrics.csv","\n".join([",".join(rows[0].keys())]+[",".join(json.dumps(row.get(key,""),ensure_ascii=False) for key in rows[0].keys()) for row in rows]) if rows else "")
        art.status("success","run",datasets=[result["dataset"] for result in all_results],decision_threshold=cfg["optimization"]["threshold"]); print(art.path); return 0
    except Exception as exc:
        art.write_text("error.traceback.txt",traceback.format_exc()); art.status("failed",args.mode,error_type=type(exc).__name__,message=str(exc),recoverable=True); print(f"运行失败，详情: {art.path}",file=sys.stderr); return 5


if __name__ == "__main__": raise SystemExit(main())
