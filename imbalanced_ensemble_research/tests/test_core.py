import math
import unittest
from unittest.mock import patch
from src.imbalance_research.core import BaseLearnerSpec, Candidate, GenerationPoolManifest, assert_binding, complexity, decode, metric_record, promethee, retrain_selected_candidate

class CoreTests(unittest.TestCase):
    def specs(self, uid="pool-a"):
        return tuple(BaseLearnerSpec(uid,1,j,f"c{j}",[.9,.1] if j==0 else [.1,.9],[.9,.1,.1,.1] if j==0 else [.1,.1,.9,.1],j,"fh","th",{"C":1,"gamma":"scale","probability":True,"tol":.001,"max_iter":-1},[]) for j in range(2))
    def candidate(self, uid="pool-a"):
        return Candidate("selected",1,0,[.1,.1],[.1]*4,[.9,.9],[.4,.6],pool_uid=uid)
    def test_empty_decode_and_class_coverage(self):
        selected, repairs=decode([.1,.2,.2],"f"); self.assertEqual(selected,[1]); self.assertEqual(repairs[0]["index"],1)
        selected, repairs=decode([.1,.2,.3,.4],"i",[0,1,0,1]); self.assertEqual(selected,[0,3]); self.assertTrue(any(r["kind"]=="class_coverage" for r in repairs))
    def test_metrics_and_complexity(self):
        metric=metric_record([0,0,1,1],[0,1,0,1]); self.assertEqual(metric["confusion_matrix"],[[1,1],[1,1]]); self.assertEqual(metric["FPR"],.5); self.assertEqual(metric["FNR"],.5)
        c=Candidate("x",0,0,[.9,.9],[.9,.9,.9,.9],[.9,.1],[.5,.5]); self.assertTrue(0 <= complexity(c,2,4,2,[.35,.35,.2,.1]) <= 1)
    def test_cross_pool_rejected(self):
        manifest=GenerationPoolManifest("pool-a",1,self.specs(),"fh","th")
        with self.assertRaisesRegex(ValueError,"跨池"): assert_binding(self.candidate("pool-b"),manifest)
    def test_retrain_preserves_each_slot_specification(self):
        manifest=GenerationPoolManifest("pool-a",1,self.specs(),"fh","th"); selected=self.candidate()
        calls=[]
        def fake_fit(spec,*args): calls.append(spec); return [0.2,0.8]
        with patch("src.imbalance_research.core.train_and_predict",side_effect=fake_fit):
            scores, info=retrain_selected_candidate(selected,manifest,[[1,2]]*4,[0,1,0,1],[[2,3],[4,5]])
        self.assertEqual([s.f for s in calls],[[.9,.1],[.1,.9]])
        self.assertEqual([s.i for s in calls],[[.9,.1,.1,.1],[.1,.1,.9,.1]])
        self.assertEqual(len(info["slot_specs"]),2); self.assertEqual(list(scores),[.2,.8])
    def test_promethee_minimize(self):
        a=self.candidate(); a.uid="a"; a.valid=True; a.objectives=(.1,.1,.2)
        b=self.candidate(); b.uid="b"; b.valid=True; b.objectives=(.8,.8,.8)
        d=promethee([a,b],{"q":.01,"p":.05,"weights":[.45,.45,.1]}); self.assertEqual(d["selected_uid"],"a")
if __name__ == "__main__": unittest.main()
