from __future__ import annotations
import json
from pathlib import Path
import pytest
from rag_practice.evaluation.multimodal import evaluate_multimodal_system
from rag_practice.multimodal import MultimodalImageIndex, read_p3_ppm
ROOT=Path(__file__).resolve().parents[1]; BENCHMARK=ROOT/"benchmarks"/"m08_multimodal"
def _index(): return MultimodalImageIndex(BENCHMARK)
def _queries(): return [json.loads(line) for line in (BENCHMARK/"queries.jsonl").read_text().splitlines() if line.strip()]
def test_ppm_parser_recovers_pixel_geometry():
    image=read_p3_ppm(BENCHMARK/"images"/"p4.ppm"); assert (image.width,image.height,image.max_value)==(8,8,255); assert image.color_count("red")==4; assert image.dominant_quadrant("red")=="upper-left"
def test_text_surrogate_cannot_observe_omitted_visual_state():
    index=_index(); query="Which control panel has a red indicator in the upper-left?"; result=index.retrieve_text(query,k=3); assert result.ranking[0][0] != "p4"; assert result.evidence_modalities==("text_surrogate",); assert index.answer(query,result,allow_text=True,allow_pixels=False)=="UNSUPPORTED_VISUAL_EVIDENCE"
def test_pixel_native_solves_visual_only_but_not_cross_modal_identity():
    index=_index(); visual=index.retrieve_pixels("Which control panel has a red indicator in the upper-left?",k=3); cross=index.retrieve_pixels("For the beta pump panel, where is the red indicator?",k=3); assert visual.ranking[0][0]=="p4"; assert cross.ranking[0][0] != "p5"; assert cross.visual_candidates_scored==9
def test_multimodal_fusion_combines_identity_and_pixel_evidence():
    index=_index(); query="For the beta pump panel, where is the red indicator?"; result=index.retrieve_multimodal(query,k=3); assert result.ranking[0][0]=="p5"; assert result.evidence_modalities==("text_surrogate","image"); assert result.visual_candidates_scored==2; assert index.answer(query,result,allow_text=True,allow_pixels=True)=="lower-right"
def test_multimodal_fusion_abstains_when_visual_constraint_has_no_evidence():
    index=_index(); query="For the beta pump panel, where is the yellow indicator?"; result=index.retrieve_multimodal(query,k=3); assert result.ranking==(); assert index.answer(query,result,allow_text=True,allow_pixels=True)=="NO_EVIDENCE"
def test_shared_source_contract_preserves_image_provenance():
    hit=_index().search("Which evacuation diagram places the yellow marker in the lower-left?",limit=1)[0]; assert hit.record.id=="d2"; assert hit.record.source_type=="image"; assert hit.record.locator=="image://benchmark/d2.ppm"; assert hit.details["evidence_modalities"]==("text_surrogate","image")
def test_evaluation_keeps_retrieval_visual_grounding_and_answer_quality_separate():
    index=_index(); queries=_queries(); expected={"text_surrogate":{"recall@3":.875,"hit_rate@1":.5,"no_evidence_accuracy":0.,"answer_correct_rate":.2,"visual_evidence_grounded_rate":0.},"pixel_native":{"recall@3":.625,"hit_rate@1":.5,"no_evidence_accuracy":0.,"answer_correct_rate":.5,"visual_evidence_grounded_rate":2/3},"multimodal_fusion":{"recall@3":1.,"hit_rate@1":1.,"no_evidence_accuracy":1.,"answer_correct_rate":1.,"visual_evidence_grounded_rate":1.}}
    for name,retrieve,allow_text,allow_pixels in (("text_surrogate",index.retrieve_text,True,False),("pixel_native",index.retrieve_pixels,False,True),("multimodal_fusion",index.retrieve_multimodal,True,True)):
        rankings={}; answers={}; modalities={}; candidates={}
        for query in queries:
            result=retrieve(str(query["query"]),k=3); qid=str(query["id"]); rankings[qid]=[item[0] for item in result.ranking]; answers[qid]=index.answer(str(query["query"]),result,allow_text=allow_text,allow_pixels=allow_pixels); modalities[qid]=result.evidence_modalities; candidates[qid]=result.visual_candidates_scored
        metrics=evaluate_multimodal_system(queries,rankings=rankings,answers=answers,evidence_modalities=modalities,visual_candidates_scored=candidates)
        for metric,value in expected[name].items(): assert metrics[metric]==pytest.approx(value)
