from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import sys
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RECOMMENDED = {"numpy": "1.26.4", "pandas": "2.2.3", "scikit-learn": "1.5.2", "matplotlib": "3.9.2", "imbalanced-learn": "0.12.4", "pypdf": "4.3.1"}


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if hasattr(value, "tolist"): return value.tolist()
    if isinstance(value, tuple): return list(value)
    raise TypeError(type(value).__name__)


def _configured_columns(value: Any, key: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(column, str) or not column for column in value):
        raise ValueError(f"preprocessing.{key} 必须是非空列名组成的列表（可为空列表）")
    if len(set(value)) != len(value):
        raise ValueError(f"preprocessing.{key} 包含重复列名")
    return value


def validate_preprocessing_columns(feature_names: list[str], preprocessing: dict) -> dict:
    """Validate the complete raw-column policy before train-only fitting."""
    required = ("drop_columns", "categorical_columns", "identifier_columns", "numeric_columns")
    if not isinstance(preprocessing, dict) or any(key not in preprocessing for key in required):
        raise ValueError("preprocessing 必须声明 drop_columns、categorical_columns、identifier_columns 和 numeric_columns")
    groups = {key: _configured_columns(preprocessing[key], key) for key in required}
    configured = [column for columns in groups.values() for column in columns]
    if len(set(configured)) != len(configured):
        raise ValueError("预处理列不能同时属于多个处理组")
    unknown = sorted(set(configured) - set(feature_names))
    if unknown:
        raise ValueError(f"预处理配置包含不存在的特征列: {unknown}")
    retained = set(feature_names) - set(groups["drop_columns"]) - set(groups["identifier_columns"])
    classified = set(groups["numeric_columns"]) | set(groups["categorical_columns"])
    if retained != classified:
        missing = sorted(retained - classified); extra = sorted(classified - retained)
        raise ValueError(f"每个保留特征必须恰好声明为 numeric 或 categorical; 缺少={missing}, 非保留={extra}")
    return {"raw_features": list(feature_names), "configured": groups, "retained_features": [x for x in feature_names if x in retained]}


def load_config(path: str | Path) -> tuple[dict, Path]:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    for section in ("inputs", "outputs", "experiment", "datasets", "preprocessing", "models", "optimization", "promethee"):
        if section not in raw: raise ValueError(f"配置缺少 {section}")
    experiment = raw["experiment"]
    if experiment["split"] != [0.7, 0.2, 0.1] or not experiment["seeds"]:
        raise ValueError("split 必须为 [0.7,0.2,0.1] 且 seeds 非空")
    profile = experiment.get("profile")
    if profile not in ("smoke", "formal", "formal_template"):
        raise ValueError("experiment.profile 必须为 smoke、formal 或 formal_template")
    opt = raw["optimization"]
    if opt["population_size"] < 4 or opt["K"] != opt["population_size"] or opt["generations"] < 1:
        raise ValueError("population_size>=4、K 必须等于 population_size、generations>=1")
    threshold = opt.get("threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("optimization.threshold 必须属于 [0,1]")
    if not 0 <= opt["CR"] <= 1 or opt["F"] <= 0 or len(opt["complexity_weights"]) != 4 or not math.isclose(sum(opt["complexity_weights"]), 1.0):
        raise ValueError("优化参数或复杂度权重无效")
    if profile == "smoke" and (opt["population_size"], opt["generations"]) != (4, 2):
        raise ValueError("smoke 配置必须使用 N=K=4、generations=2")
    if profile in ("formal", "formal_template") and (opt["population_size"], opt["generations"]) != (20, 100):
        raise ValueError("formal 配置必须使用 N=K=20、generations=100")
    if profile == "formal_template" and (not raw.get("requires_user_label_semantics_confirmation") or raw["datasets"]):
        raise ValueError("formal_template 必须标记标签语义待确认且不含数据集")
    for key in ("drop_columns", "categorical_columns", "identifier_columns", "numeric_columns"):
        _configured_columns(raw["preprocessing"].get(key), key)
    prom = raw["promethee"]
    if not (0 <= prom["q"] < prom["p"] <= 1) or len(prom["weights"]) != 3 or not math.isclose(sum(prom["weights"]), 1.0):
        raise ValueError("PROMETHEE 参数无效")
    base = path.parent
    for key in ("data_dir", "reference_pdf"):
        raw["inputs"][key] = str((base / raw["inputs"][key]).resolve()) if not Path(raw["inputs"][key]).is_absolute() else raw["inputs"][key]
    raw["outputs"]["root"] = str((base / raw["outputs"]["root"]).resolve())
    for ds in raw["datasets"]:
        for needed in ("id", "path", "format", "label_column", "positive_label"):
            if needed not in ds: raise ValueError(f"数据集配置缺少 {needed}")
        ds["path"] = str((base / ds["path"]).resolve()) if not Path(ds["path"]).is_absolute() else ds["path"]
    return raw, path


class Artifacts:
    def __init__(self, config: dict, command: str, mode: str):
        root = Path(config["outputs"]["root"]).resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{mode}_{uuid.uuid4().hex[:8]}"
        self.path = root / self.run_id
        self.path.mkdir()
        self.write_text("command.txt", command + "\n")
        self.write_json("config.snapshot.json", config)
    def write_json(self, name: str, value: Any):
        self.write_text(name, json.dumps(value, ensure_ascii=False, indent=2, default=jsonable) + "\n")
    def write_text(self, name: str, content: str):
        target = self.path / name; target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(content, encoding="utf-8"); os.replace(temp, target)
    def status(self, state: str, stage: str, **extra):
        self.write_json("status.json", {"run_id": self.run_id, "status": state, "stage": stage, "timestamp_utc": datetime.now(timezone.utc).isoformat(), **extra})


def fingerprint(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists(): return {"path": str(p), "exists": False}
    s = p.stat()
    if p.is_dir():
        return {"path": str(p.resolve()), "exists": True, "kind": "directory", "mtime_ns": s.st_mtime_ns}
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return {"path": str(p.resolve()), "exists": True, "kind": "file", "bytes": s.st_size, "mtime_ns": s.st_mtime_ns, "sha256": h.hexdigest()}


def environment_report() -> dict:
    packages = {}
    for package, suggested in RECOMMENDED.items():
        try: packages[package] = {"available": True, "version": importlib.metadata.version(package), "suggested_fixed_version": suggested}
        except importlib.metadata.PackageNotFoundError: packages[package] = {"available": False, "version": None, "suggested_fixed_version": suggested, "blocked_function": {"imbalanced-learn": "RUSBoost", "pypdf": "PDF 文本/格式提取", "matplotlib": "图表", "numpy": "训练", "pandas": "表格处理", "scikit-learn": "拆分、预处理和模型"}[package]}
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "packages": packages, "network_used": False, "installation_performed": False}


def parse_keel(path: str | Path) -> tuple[list[str], list[list[str]], list[dict]]:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    columns, attrs, data_start = [], [], None
    for n, line in enumerate(lines):
        m = re.match(r"\s*@attribute\s+([^\s]+)\s+(.+)", line, re.I)
        if m:
            columns.append(m.group(1)); attrs.append({"name": m.group(1), "declaration": m.group(2).strip()})
        if re.match(r"\s*@data\s*$", line, re.I): data_start = n + 1; break
    if data_start is None or not columns: raise ValueError("未识别 KEEL @attribute/@data")
    rows = [[x.strip() for x in r] for r in csv.reader(lines[data_start:], skipinitialspace=True) if r and any(x.strip() for x in r)]
    if any(len(r) != len(columns) for r in rows): raise ValueError("数据行列数与属性声明不一致")
    return columns, rows, attrs


def audit_dataset(path: Path) -> dict:
    report = {"path": str(path), "file": path.name, "format": path.suffix.lower(), "fingerprint": fingerprint(path)}
    try:
        if path.suffix.lower() != ".dat": raise ValueError("当前审计仅实现 .dat/KEEL")
        cols, rows, attrs = parse_keel(path)
        values = list(zip(*rows)) if rows else []
        types, missing, nonnumeric, constants, candidates = {}, {}, [], [], []
        for i, (name, col) in enumerate(zip(cols, values)):
            miss = sum(v in ("", "?", "NA", "NaN") for v in col); missing[name] = miss
            numeric = 0
            for v in col:
                try: float(v); numeric += 1
                except ValueError: pass
            types[name] = "numeric" if numeric + miss == len(col) else "non_numeric"
            if types[name] == "non_numeric": nonnumeric.append(name)
            if len(set(col)) <= max(10, len(rows) // 20): candidates.append({"column": name, "unique": len(set(col)), "reason": "低基数，仅为候选"})
            if len(set(col)) <= 1: constants.append(name)
        label = cols[-1] if cols else None
        counts = {v: values[-1].count(v) for v in sorted(set(values[-1]))} if values else {}
        report.update({"parse_status": "success", "parser": "KEEL @attribute/@data", "rows": len(rows), "columns": cols, "dimensions": [len(rows), len(cols)], "raw_feature_count": max(0, len(cols)-1), "candidate_label_columns": candidates, "observed_last_column": label, "last_column_counts": counts, "last_column_ratios": {k: v / len(rows) for k,v in counts.items()} if rows else {}, "column_types": types, "missing_by_column": missing, "non_numeric_columns": nonnumeric, "constant_columns": constants, "duplicate_rows": len(rows)-len({tuple(x) for x in rows}), "duplicate_feature_rows": len(rows)-len({tuple(x[:-1]) for x in rows}), "attributes": attrs, "leakage_risks": ["标签语义、标签派生字段、近重复跨拆分、时间/组依赖和标识符不能仅凭静态文件排除；需人工核验。"]})
    except Exception as exc: report.update({"parse_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    return report


def audit_inputs(config: dict) -> dict:
    data_dir, pdf = Path(config["inputs"]["data_dir"]), Path(config["inputs"]["reference_pdf"])
    files = sorted([p for p in data_dir.glob("*") if p.is_file()]) if data_dir.is_dir() else []
    pdf_info = {"path": str(pdf), "exists": pdf.exists(), "fingerprint": fingerprint(pdf)}
    if pdf.exists():
        try:
            import pypdf
            reader = pypdf.PdfReader(str(pdf)); text = "\n".join((p.extract_text() or "")[:300] for p in reader.pages[:3])
            pdf_info.update({"status": "success", "pages": len(reader.pages), "format_observation": {"heading_number_tokens": re.findall(r"(?m)^\s*\d+(?:\.\d+)*\s+[^\n]{0,80}", text)[:20], "table_figure_tokens": re.findall(r"(?:表|图)\s*\d+", text)[:20], "note": "仅提取短编号模式和结构线索，未复制正文。"}})
        except Exception as exc: pdf_info.update({"status": "not_extracted", "reason": f"{type(exc).__name__}: {exc}", "suggested_package": "pypdf==4.3.1"})
    else: pdf_info.update({"status": "missing", "reason": "指定路径不存在"})
    return {"input_boundary": [str(data_dir), str(pdf)], "data_dir_exists": data_dir.is_dir(), "data_files": [audit_dataset(p) for p in files], "reference_pdf": pdf_info}


def load_dataset(spec: dict):
    if spec["format"] != "keel_dat": raise ValueError("当前运行仅支持 keel_dat")
    p = Path(spec["path"])
    if not p.is_file(): raise ValueError(f"数据文件不存在: {p}")
    cols, rows, attrs = parse_keel(p)
    label = spec["label_column"]
    if label not in cols: raise ValueError(f"标签列不存在: {label}; 可用列: {cols}")
    li = cols.index(label); labels = [r[li] for r in rows]
    unique = sorted(set(labels))
    if len(unique) != 2: raise ValueError(f"必须二分类，实际类别: {unique}")
    if spec["positive_label"] not in unique: raise ValueError(f"positive_label 未匹配: {spec['positive_label']}; 实际: {unique}")
    features = [c for c in cols if c != label]
    X = [[r[cols.index(c)] for c in features] for r in rows]
    y = [1 if x == spec["positive_label"] else 0 for x in labels]
    return X, y, features, {"columns": cols, "label_mapping": {"positive_original": spec["positive_label"], "negative_original": next(x for x in unique if x != spec["positive_label"])}, "attributes": attrs}


def metric_record(y_true, y_pred, scores=None, threshold=None) -> dict:
    tn = fp = fn = tp = 0
    for y, p in zip(y_true, y_pred):
        if y == 0 and p == 0: tn += 1
        elif y == 0 and p == 1: fp += 1
        elif y == 1 and p == 0: fn += 1
        elif y == 1 and p == 1: tp += 1
    def safe(a,b): return None if b == 0 else a/b
    fpr, fnr, precision, recall, tnr = safe(fp,fp+tn), safe(fn,fn+tp), safe(tp,tp+fp), safe(tp,tp+fn), safe(tn,tn+fp)
    f1 = None if precision is None or recall is None or precision+recall == 0 else 2*precision*recall/(precision+recall)
    gmean = None if recall is None or tnr is None else math.sqrt(recall*tnr)
    out = {"confusion_matrix": [[tn,fp],[fn,tp]], "FPR": fpr, "FNR": fnr, "precision": precision, "recall": recall, "F1": f1, "G_mean": gmean, "ROC_AUC": None, "PR_AUC": None, "not_computable_reason": []}
    if threshold is not None: out["decision_threshold"] = threshold
    if scores is not None and len(set(y_true)) == 2:
        try:
            ordered=sorted(zip(scores,y_true), key=lambda pair: pair[0])
            positive_count=sum(y_true); negative_count=len(y_true)-positive_count
            ranks=[]; position=0
            while position < len(ordered):
                end=position+1
                while end < len(ordered) and ordered[end][0] == ordered[position][0]: end += 1
                ranks.extend([(position+1+end)/2]*(end-position)); position=end
            positive_rank_sum=sum(rank for rank,(_,label) in zip(ranks,ordered) if label == 1)
            out["ROC_AUC"]=(positive_rank_sum-positive_count*(positive_count+1)/2)/(positive_count*negative_count)
            descending=sorted(zip(scores,y_true), key=lambda pair: pair[0], reverse=True)
            true_positive=false_positive=0; average_precision=0.0; position=0
            while position < len(descending):
                end=position+1
                while end < len(descending) and descending[end][0] == descending[position][0]: end += 1
                positives=sum(label for _,label in descending[position:end]); true_positive += positives; false_positive += end-position-positives
                average_precision += (true_positive/(true_positive+false_positive))*positives/positive_count; position=end
            out["PR_AUC"]=average_precision
        except Exception as exc: out["not_computable_reason"].append(f"AUC: {type(exc).__name__}")
    else: out["not_computable_reason"].append("无连续得分或测试集不含两类")
    return out


@dataclass(frozen=True)
class BaseLearnerSpec:
    pool_uid: str; generation: int; slot: int; chromosome_uid: str; f: list[float]; i: list[float]; derived_seed: int; feature_order_hash: str; train_index_hash: str; svm_params: dict; repairs: list[dict]
@dataclass(frozen=True)
class GenerationPoolManifest:
    pool_uid: str; generation: int; slots: tuple[BaseLearnerSpec, ...]; feature_order_hash: str; train_index_hash: str
@dataclass
class Candidate:
    uid: str; generation: int; slot: int; f: list[float]; i: list[float]; m: list[float]; w: list[float]; pool_uid: str | None = None; objectives: tuple[float,float,float] | None = None; valid: bool = False; invalid_reason: str | None = None; manifest: GenerationPoolManifest | None = None


def decode(values: list[float], kind: str, y=None) -> tuple[list[int], list[dict]]:
    selected = [j for j,v in enumerate(values) if v >= .5]; repairs=[]
    if not selected:
        best=max(values); selected=[next(j for j,v in enumerate(values) if v==best)]; repairs.append({"kind":"empty_selection_argmax","segment":kind,"index":selected[0]})
    if kind == "i" and y is not None:
        for label in (0,1):
            if not any(y[j] == label for j in selected):
                available=[j for j,x in enumerate(y) if x==label]
                if not available: raise ValueError(f"训练集不存在类别 {label}")
                selected.append(min(available)); repairs.append({"kind":"class_coverage","label":label,"index":min(available)})
    return sorted(set(selected)), repairs

def complexity(candidate: Candidate, D: int, T: int, K: int, weights: list[float]) -> float:
    f,_=decode(candidate.f,"f"); i,_=decode(candidate.i,"i"); m,_=decode(candidate.m,"m")
    raw=[candidate.w[j] for j in m]; total=sum(raw); normalized=[x/total for x in raw] if total else [1/len(m)]*len(m)
    entropy_term=1.0 if len(m)==1 else 1 - (-sum(x*math.log(x) for x in normalized if x>0)/math.log(len(m)))
    return weights[0]*len(f)/D + weights[1]*len(i)/T + weights[2]*len(m)/K + weights[3]*entropy_term

def assert_binding(candidate: Candidate, manifest: GenerationPoolManifest):
    if candidate.pool_uid != manifest.pool_uid: raise ValueError("pool_uid 不匹配，拒绝跨池 m/w")
    if len(candidate.m) != len(manifest.slots) or len(candidate.w) != len(manifest.slots): raise ValueError("m/w 长度与模型池不一致")
    if any(s.pool_uid != manifest.pool_uid for s in manifest.slots): raise ValueError("manifest 槽位 pool_uid 不一致")

def promethee(candidates: list[Candidate], cfg: dict) -> dict:
    valid=[c for c in candidates if c.valid and c.objectives and all(math.isfinite(v) for v in c.objectives)]
    if not valid: return {"status":"no_candidate"}
    if len(valid)==1: return {"status":"single_candidate_fallback","selected_uid":valid[0].uid,"ranking":[valid[0].uid]}
    q,p,weights=cfg["q"],cfg["p"],cfg["weights"]
    def pref(a,b):
        return sum(w*(0 if b[j]-a[j] <= q else 1 if b[j]-a[j] >= p else (b[j]-a[j]-q)/(p-q)) for j,w in enumerate(weights))
    flows=[]
    for a in valid:
        plus=sum(pref(a.objectives,b.objectives) for b in valid if b is not a)/(len(valid)-1); minus=sum(pref(b.objectives,a.objectives) for b in valid if b is not a)/(len(valid)-1)
        flows.append({"uid":a.uid,"positive":plus,"negative":minus,"net":plus-minus})
    rank=sorted(flows,key=lambda r:(-r["net"], next(c.objectives for c in valid if c.uid==r["uid"]),r["uid"]))
    return {"status":"success","direction":"all_minimize","preference":"V-shape","q":q,"p":p,"weights":weights,"flows":flows,"ranking":[r["uid"] for r in rank],"selected_uid":rank[0]["uid"]}


def nondominated(candidates: list[Candidate]) -> list[Candidate]:
    valid=[c for c in candidates if c.valid and c.objectives]
    return [a for a in valid if not any(all(x<=y for x,y in zip(b.objectives,a.objectives)) and any(x<y for x,y in zip(b.objectives,a.objectives)) for b in valid if b is not a)]


def train_and_predict(spec: BaseLearnerSpec, X_train, y_train, X_eval):
    import numpy as np
    from sklearn.svm import SVC
    fs,_=decode(spec.f,"f"); ii,_=decode(spec.i,"i",y_train)
    model=SVC(kernel="rbf", random_state=spec.derived_seed, **spec.svm_params)
    model.fit(np.asarray(X_train)[ii][:,fs], np.asarray(y_train)[ii])
    return model.predict_proba(np.asarray(X_eval)[:,fs])[:,list(model.classes_).index(1)]

def retrain_selected_candidate(selected: Candidate, pool: GenerationPoolManifest, X_train, y_train, X_test):
    """按绑定清单逐槽重训，绝不把 selected.f/i 重复用于 K 槽。"""
    assert_binding(selected,pool)
    scores=[]; unavailable=[]
    for spec in pool.slots:
        try: scores.append(train_and_predict(spec,X_train,y_train,X_test))
        except Exception as exc: scores.append(None); unavailable.append({"slot":spec.slot,"reason":type(exc).__name__})
    chosen,_=decode(selected.m,"m")
    if any(scores[j] is None for j in chosen): raise ValueError(f"所选槽位 unavailable: {unavailable}")
    raw=[selected.w[j] for j in chosen]; total=sum(raw); ws=[x/total for x in raw] if total else [1/len(chosen)]*len(chosen)
    import numpy as np
    return np.sum([w*np.asarray(scores[j], dtype=float) for j,w in zip(chosen,ws)],axis=0), {"final_pool_uid": stable_hash({"source":pool.pool_uid,"selected":selected.uid,"phase":"final"}), "source_pool_uid":pool.pool_uid,"slot_specs": [asdict(s) for s in pool.slots], "unavailable":unavailable}
