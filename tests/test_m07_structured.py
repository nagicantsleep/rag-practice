import json
from pathlib import Path
from rag_practice.structured import (
    DualLevelGraphRetriever, GlobalGraphRetriever, HippoRAGRetriever, KAGPathRetriever, KnowledgeGraph,
    MemoryEvent, RaptorStyleIndex, StructuredDocument, TemporalMemoryIndex,
)

BASE = Path(__file__).parents[1] / "benchmarks" / "m07_structured"

def load(name): return [json.loads(line) for line in (BASE/name).read_text().splitlines() if line.strip()]
def docs(): return [StructuredDocument.from_mapping(r) for r in load("documents.jsonl")]

def test_raptor_style_routes_collection_wide_research_evidence():
    index=RaptorStyleIndex(docs())
    ranking=[d for d,_ in index.search("What does Atlas Network labs studies across the collection?",k=3)]
    assert set(ranking)=={"d2","d6","d10"}

def test_raptor_style_returns_empty_when_routed_group_has_no_leaf_match():
    index=RaptorStyleIndex(docs())
    assert index.search("Which countries host Atlas Network labs?",k=6)==[]

def test_kag_path_retrieves_complete_three_hop_currency_chain():
    retriever=KAGPathRetriever(KnowledgeGraph(docs()))
    ranking=[d for d,_ in retriever.search("Which currency is used in the country containing Aurora Lab?",k=5)]
    assert ranking==["d1","d3","d4"]

def test_graph_global_collects_all_atlas_country_evidence():
    retriever=GlobalGraphRetriever(KnowledgeGraph(docs()))
    ranking=[d for d,_ in retriever.search("Which countries host Atlas Network labs?",k=6)]
    assert set(ranking)=={"d1","d3","d5","d7","d9","d11"}

def test_light_rag_dual_level_routes_local_and_global_without_task_labels():
    retriever=DualLevelGraphRetriever(KnowledgeGraph(docs()))
    assert retriever.route_level("Which currency is used in the country containing Vega Lab?")=="low"
    assert retriever.route_level("Which countries host Atlas Network labs?")=="high"
    low=[d for d,_ in retriever.search("Which currency is used in the country containing Vega Lab?",k=3)]
    high=[d for d,_ in retriever.search("Which countries host Atlas Network labs?",k=6)]
    assert low==["d5","d7","d8"]
    assert set(high)=={"d1","d3","d5","d7","d9","d11"}

def test_hipporag_multi_seed_bridge_prioritizes_association_path():
    retriever=HippoRAGRetriever(KnowledgeGraph(docs()))
    ranking=[d for d,_ in retriever.search("Which Atlas Network lab is connected to euro?",k=3)]
    assert set(ranking)=={"d5","d7","d8"}

def test_temporal_memory_returns_latest_and_previous_versions():
    base=[MemoryEvent.from_mapping(r) for r in load("memory.jsonl")]
    index=TemporalMemoryIndex(base)
    for row in load("memory_updates.jsonl"): index.add(MemoryEvent.from_mapping(row))
    assert index.search("What database version does Vega Lab currently use?",k=1)[0][0]=="mem4"
    assert index.search("What database version did Vega Lab use before the current version?",k=1)[0][0]=="mem1"
