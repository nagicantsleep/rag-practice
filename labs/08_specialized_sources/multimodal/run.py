from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter_ns
from rag_practice.evaluation.multimodal import evaluate_multimodal_system
from rag_practice.multimodal import MultimodalImageIndex
ROOT=Path(__file__).resolve().parents[3]; BENCHMARK=ROOT/"benchmarks"/"m08_multimodal"; RESULTS=Path(__file__).resolve().parent/"results"
def load_queries(): return [json.loads(line) for line in (BENCHMARK/"queries.jsonl").read_text().splitlines() if line.strip()]
def evaluate_system(index,queries,*,name):
    if name=="text_surrogate": retrieve=index.retrieve_text; allow_text,allow_pixels=True,False
    elif name=="pixel_native": retrieve=index.retrieve_pixels; allow_text,allow_pixels=False,True
    elif name=="multimodal_fusion": retrieve=index.retrieve_multimodal; allow_text,allow_pixels=True,True
    else: raise ValueError(name)
    rankings={}; answers={}; modalities={}; visual_candidates={}; per_query=[]
    for query in queries:
        qid=str(query["id"]); started=perf_counter_ns(); result=retrieve(str(query["query"]),k=3); latency_ms=(perf_counter_ns()-started)/1_000_000; ranking=[item[0] for item in result.ranking]; answer=index.answer(str(query["query"]),result,allow_text=allow_text,allow_pixels=allow_pixels)
        rankings[qid]=ranking; answers[qid]=answer; modalities[qid]=list(result.evidence_modalities); visual_candidates[qid]=result.visual_candidates_scored
        per_query.append({"id":qid,"task":query["task"],"query":query["query"],"relevant":query["relevant"],"ranking":ranking,"top_locator":index.by_id[ranking[0]].locator if ranking else None,"answer":answer,"expected_answer":query["answer"],"answer_correct":answer==query["answer"],"evidence_modalities":list(result.evidence_modalities),"visual_candidates_scored":result.visual_candidates_scored,"latency_ms":latency_ms})
    metrics=evaluate_multimodal_system(queries,rankings=rankings,answers=answers,evidence_modalities=modalities,visual_candidates_scored=visual_candidates); metrics["mean_query_ms"]=sum(float(item["latency_ms"]) for item in per_query)/len(per_query); return {"metrics":metrics,"per_query":per_query}
def render_markdown(result):
    lines=["# M08.5 Multimodal RAG results","","Benchmark: 9 raster images, 10 queries (visual-only, text-sufficient, cross-modal, no-evidence).","","| System | Recall@3 | Hit@1 | Visual Hit@1 | Cross-modal Hit@1 | Text Hit@1 | No-evidence | Answer correct | Visual grounded | Visual candidates |","| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in ("text_surrogate","pixel_native","multimodal_fusion"):
        m=result["systems"][name]["metrics"]; lines.append(f"| {name} | {m['recall@3']:.3f} | {m['hit_rate@1']:.3f} | {m['visual_required_hit@1']:.3f} | {m['cross_modal_hit@1']:.3f} | {m['text_sufficient_hit@1']:.3f} | {m['no_evidence_accuracy']:.3f} | {m['answer_correct_rate']:.3f} | {m['visual_evidence_grounded_rate']:.3f} | {m['mean_visual_candidates_scored']:.1f} |")
    lines += ["","## Interpretation guardrails","","- Text-surrogate retrieval never inspects pixels; a caption hit is not visual evidence.","- Pixel-native retrieval deliberately ignores asset/site metadata, so it can solve visual-only questions yet fail cross-modal identity constraints.","- Multimodal fusion applies explicit site/kind constraints, then combines BM25 surrogate relevance with pixel evidence when the query asks for a visual property.","- Answer correctness is reported separately from retrieval: a wrong image can accidentally yield the same short answer string.","- The raster parser and visual features are deterministic teaching controls, not a learned vision model."]
    return "\n".join(lines)+"\n"
def main():
    index=MultimodalImageIndex(BENCHMARK); queries=load_queries(); result={"hypothesis":"captions and pixels are complementary evidence: text surrogates cannot support visual-only claims, while image-only matching loses identity/context constraints","benchmark":{"images":len(index.assets),"queries":len(queries),"text_chars_indexed":index.text_chars_indexed,"image_bytes_indexed":index.image_bytes_indexed},"systems":{}}
    for name in ("text_surrogate","pixel_native","multimodal_fusion"): result["systems"][name]=evaluate_system(index,queries,name=name)
    RESULTS.mkdir(parents=True,exist_ok=True); (RESULTS/"results.json").write_text(json.dumps(result,indent=2)+"\n"); markdown=render_markdown(result); (RESULTS/"results.md").write_text(markdown); print(markdown,end="")
if __name__=="__main__": main()
