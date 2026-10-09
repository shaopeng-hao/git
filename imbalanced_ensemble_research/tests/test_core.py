import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.imbalance_research.cli import evaluate_pool, prepare_numeric, run_one, split_indices
from src.imbalance_research.core import BaseLearnerSpec, Candidate, GenerationPoolManifest, assert_binding, complexity, decode, load_config, metric_record, promethee, retrain_selected_candidate


class CoreTests(unittest.TestCase):
    def specs(self, uid="pool-a"):
        return tuple(BaseLearnerSpec(uid,1,j,f"c{j}",[.9,.1] if j==0 else [.1,.9],[.9,.1,.1,.1] if j==0 else [.1,.1,.9,.1],j,"fh","th",{"C":1,"gamma":"scale","probability":True,"tol":.001,"max_iter":-1},[]) for j in range(2))

    def candidate(self, uid="pool-a"):
        return Candidate("selected",1,0,[.1,.1],[.1]*4,[.9,.9],[.4,.6],pool_uid=uid)

    def test_empty_decode_and_class_coverage(self):
        selected, repairs=decode([.1,.2,.2],"f"); self.assertEqual(selected,[1]); self.assertEqual(repairs[0]["index"],1)
        selected, repairs=decode([.1,.2,.3,.4],"i",[0,1,0,1]); self.assertEqual(selected,[0,3]); self.assertTrue(any(repair["kind"]=="class_coverage" for repair in repairs))

    def test_metrics_and_complexity(self):
        metric=metric_record([0,0,1,1],[0,1,0,1]); self.assertEqual(metric["confusion_matrix"],[[1,1],[1,1]]); self.assertEqual(metric["FPR"],.5); self.assertEqual(metric["FNR"],.5)
        candidate=Candidate("x",0,0,[.9,.9],[.9,.9,.9,.9],[.9,.1],[.5,.5]); self.assertTrue(0 <= complexity(candidate,2,4,2,[.35,.35,.2,.1]) <= 1)

    def test_cross_pool_rejected(self):
        manifest=GenerationPoolManifest("pool-a",1,self.specs(),"fh","th")
        with self.assertRaisesRegex(ValueError,"跨池"): assert_binding(self.candidate("pool-b"),manifest)

    def test_retrain_preserves_each_slot_specification(self):
        manifest=GenerationPoolManifest("pool-a",1,self.specs(),"fh","th"); selected=self.candidate(); calls=[]
        def fake_fit(spec,*args): calls.append(spec); return [0.2,0.8]
        with patch("src.imbalance_research.core.train_and_predict",side_effect=fake_fit):
            scores, info=retrain_selected_candidate(selected,manifest,[[1,2]]*4,[0,1,0,1],[[2,3],[4,5]])
        self.assertEqual([spec.f for spec in calls],[[.9,.1],[.1,.9]])
        self.assertEqual([spec.i for spec in calls],[[.9,.1,.1,.1],[.1,.1,.9,.1]])
        self.assertEqual(len(info["slot_specs"]),2); self.assertEqual(list(scores),[.2,.8])

    def test_promethee_minimize(self):
        first=self.candidate(); first.uid="a"; first.valid=True; first.objectives=(.1,.1,.2)
        second=self.candidate(); second.uid="b"; second.valid=True; second.objectives=(.8,.8,.8)
        decision=promethee([first,second],{"q":.01,"p":.05,"weights":[.45,.45,.1]}); self.assertEqual(decision["selected_uid"],"a")


class IntegrationTests(unittest.TestCase):
    policy={"drop_columns":[],"categorical_columns":["color"],"identifier_columns":["record_id"],"numeric_columns":["value"]}

    @staticmethod
    def candidate(uid, slot, selected_slot=0):
        return Candidate(uid,0,slot,[.9],[.9,.1,.9,.1],[.9 if i==selected_slot else .1 for i in range(4)],[.25]*4)

    def test_split_and_train_only_preprocessing(self):
        y=[0]*20+[1]*20
        train, validation, test=split_indices(y, 19)
        self.assertFalse(set(train) & set(validation)); self.assertFalse(set(train) & set(test)); self.assertFalse(set(validation) & set(test))
        self.assertEqual((len(train),len(validation),len(test)),(28,8,4))
        self.assertEqual([sum(y[i] == label for i in group) for group in (train,validation,test) for label in (0,1)],[14,14,4,4,2,2])
        Xtr=[["1","red","id-a"],["3","blue","id-b"],["?","red","id-c"]]
        Xv=[["100","green","id-v"]]; Xte=[["200","blue","id-t"]]
        train_matrix, validation_matrix, test_matrix, names, audit=prepare_numeric(Xtr,Xv,Xte,["value","color","record_id"],self.policy)
        self.assertNotIn("record_id",names); self.assertIn("color=red",names); self.assertIn("color=blue",names)
        self.assertEqual(audit["numeric_imputation"]["value"]["value"],2.0)
        self.assertEqual(audit["categorical_encoding"]["color"]["validation_unknown_count"],1)
        self.assertEqual(audit["fit_scope"],"train_only")
        self.assertAlmostEqual(float(train_matrix[:,names.index("value")].mean()),0.0)
        self.assertGreater(float(validation_matrix[0,names.index("value")]),50.0)
        self.assertGreater(float(test_matrix[0,names.index("value")]),100.0)

    def test_nondefault_threshold_changes_validation_objective(self):
        y=[0,1]
        def deterministic(*args): return [0.4,0.6]
        low=[self.candidate(f"low-{slot}",slot) for slot in range(4)]
        high=[self.candidate(f"high-{slot}",slot) for slot in range(4)]
        with patch("src.imbalance_research.cli.train_and_predict",side_effect=deterministic):
            evaluate_pool(low,[[0],[1],[0],[1]],[0,1,0,1],[[0],[1]],y,3,0,{"C":1,"gamma":"scale","probability":True,"tol":.001,"max_iter":-1},[.35,.35,.2,.1],["x"],.5)
            evaluate_pool(high,[[0],[1],[0],[1]],[0,1,0,1],[[0],[1]],y,3,0,{"C":1,"gamma":"scale","probability":True,"tol":.001,"max_iter":-1},[.35,.35,.2,.1],["x"],.7)
        self.assertEqual(low[0].objectives[:2],(0.0,0.0))
        self.assertEqual(high[0].objectives[:2],(0.0,1.0))
        self.assertEqual(metric_record(y,[0,1],[.4,.6],.5)["confusion_matrix"],[[1,0],[0,1]])
        self.assertEqual(metric_record(y,[0,0],[.4,.6],.7)["confusion_matrix"],[[1,0],[1,0]])

    def test_selected_unavailable_slot_is_invalid(self):
        candidates=[self.candidate(f"candidate-{slot}",slot,1) for slot in range(4)]
        def fail_second(spec,*args):
            if spec.slot == 1: raise RuntimeError("unavailable")
            return [.2,.8]
        with patch("src.imbalance_research.cli.train_and_predict",side_effect=fail_second):
            evaluate_pool(candidates,[[0],[1],[0],[1]],[0,1,0,1],[[0],[1]],[0,1],7,0,{"C":1,"gamma":"scale","probability":True,"tol":.001,"max_iter":-1},[.35,.35,.2,.1],["x"],.5)
        self.assertTrue(all(not candidate.valid and candidate.invalid_reason == "selected_unavailable_slot" for candidate in candidates))

    def test_run_one_audits_test_isolation_and_pool_count(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path=Path(directory)/"synthetic.dat"
            rows=[]
            for index in range(40):
                label="positive" if index >= 20 else "negative"
                rows.append(f"{index},{'high' if index % 2 else 'low'},id-{index},{label}")
            data_path.write_text("@relation synthetic\n@attribute value real\n@attribute category {low,high}\n@attribute record_id string\n@attribute class {positive,negative}\n@data\n"+"\n".join(rows),encoding="utf-8")
            config={"models":{"rbf_svm":{"C":1.0,"gamma":"scale","probability":True,"class_weight":None,"tol":.001,"max_iter":-1},"random_forest":{"n_estimators":5,"class_weight":None},"rusboost":{"n_estimators":5}},"preprocessing":{"drop_columns":[],"categorical_columns":["category"],"identifier_columns":["record_id"],"numeric_columns":["value"]},"optimization":{"enabled":True,"population_size":4,"K":4,"generations":2,"CR":.2,"F":.5,"threshold":.7,"complexity_weights":[.35,.35,.2,.1]},"promethee":{"weights":[.45,.45,.1],"q":.01,"p":.05}}
            def deterministic(spec, train_data, train_labels, evaluation_data): return [.4 if index % 2 == 0 else .8 for index in range(len(evaluation_data))]
            with patch("src.imbalance_research.cli.run_baselines",return_value=[]), patch("src.imbalance_research.cli.train_and_predict",side_effect=deterministic), patch("src.imbalance_research.core.train_and_predict",side_effect=deterministic):
                result=run_one({"id":"synthetic","path":str(data_path),"format":"keel_dat","label_column":"class","positive_label":"positive"},config,29)
        self.assertEqual(result["decision_threshold"],.7)
        self.assertEqual(result["split_summary"]["mutually_exclusive"],True)
        self.assertEqual(result["preprocessing"]["dropped_by_configuration"]["identifier_columns"],["record_id"])
        self.assertIn("category=low",result["preprocessing"]["output_features"])
        self.assertEqual(result["optimizer"]["pool_count"],3)
        self.assertEqual(len(result["optimizer"]["pool_manifests"]),1+config["optimization"]["generations"])
        self.assertEqual(result["optimizer"]["evaluation_data_scope"]["test"],"not passed to evaluate_pool or PROMETHEE")
        self.assertEqual(result["optimizer"]["decision_threshold"],.7)
        if "selected_candidate" in result["optimizer"]:
            self.assertEqual(result["optimizer"]["selected_candidate"]["test_metrics"]["decision_threshold"],.7)

    def test_configuration_threshold_profiles_and_formal_template(self):
        root=Path(__file__).resolve().parents[1]
        smoke,_=load_config(root/"configs"/"smoke.json")
        formal,_=load_config(root/"configs"/"formal.template.json")
        self.assertEqual((smoke["optimization"]["population_size"],smoke["optimization"]["generations"]),(4,2))
        self.assertEqual((formal["optimization"]["population_size"],formal["optimization"]["generations"]),(20,100))
        self.assertTrue(formal["requires_user_label_semantics_confirmation"])
        broken=json.loads(json.dumps(smoke)); broken["optimization"]["threshold"]=1.1
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"broken.json"; path.write_text(json.dumps(broken),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"threshold"): load_config(path)


if __name__ == "__main__": unittest.main()
