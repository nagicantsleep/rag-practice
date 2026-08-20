from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from rag_practice.ir.bm25 import BM25Index
from rag_practice.ir.text import tokenize
from .models import StructuredDocument

_RELATION_ALIASES = {
    "located_in": {"where", "located", "location", "host", "hosts"},
    "studies": {"study", "studies", "research", "topic", "topics"},
    "country": {"country", "countries"},
    "currency": {"currency", "currencies"},
    "includes": {"include", "includes", "network", "lab", "labs", "site", "sites"},
}

@dataclass(frozen=True)
class GraphEdge:
    subject: str
    relation: str
    object: str
    document_id: str

class KnowledgeGraph:
    def __init__(self, documents: list[StructuredDocument]):
        self.documents = {d.id: d for d in documents}
        self.edges = [GraphEdge(t.subject, t.relation, t.object, t.document_id) for d in documents for t in d.triples]
        self.entities = sorted({e.subject for e in self.edges} | {e.object for e in self.edges})
        self.forward: dict[str, list[GraphEdge]] = defaultdict(list)
        self.undirected: dict[str, list[tuple[str, GraphEdge]]] = defaultdict(list)
        for edge in self.edges:
            self.forward[edge.subject].append(edge)
            self.undirected[edge.subject].append((edge.object, edge))
            self.undirected[edge.object].append((edge.subject, edge))
        self.network_roots = sorted({e.subject for e in self.edges if e.relation == "includes" and e.subject.lower().endswith("network")})

    def match_entities(self, query: str) -> list[str]:
        q = set(tokenize(query))
        matches = []
        for entity in self.entities:
            terms = set(tokenize(entity))
            if terms and terms <= q:
                matches.append(entity)
        return sorted(matches, key=lambda e: (-len(tokenize(e)), e))

    def target_relation(self, query: str) -> str | None:
        q = set(tokenize(query))
        candidates = []
        for relation, aliases in _RELATION_ALIASES.items():
            overlap = len(q & aliases)
            if overlap:
                candidates.append((overlap, relation))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (-x[0], x[1]))
        for preferred in ("currency", "country", "located_in", "studies", "includes"):
            if any(rel == preferred for _, rel in candidates):
                return preferred
        return candidates[0][1]

    def shortest_path(self, start: str, goal: str) -> list[GraphEdge]:
        if start == goal:
            return []
        queue = deque([(start, [])]); seen = {start}
        while queue:
            node, path = queue.popleft()
            for neighbor, edge in sorted(self.undirected.get(node, []), key=lambda x: (x[0], x[1].relation, x[1].document_id)):
                if neighbor in seen:
                    continue
                next_path = path + [edge]
                if neighbor == goal:
                    return next_path
                seen.add(neighbor); queue.append((neighbor, next_path))
        return []

    def shortest_path_to_relation(self, start: str, relation: str, *, max_hops: int = 5) -> list[GraphEdge]:
        queue = deque([(start, [])]); seen = {start}
        while queue:
            node, path = queue.popleft()
            if len(path) >= max_hops:
                continue
            for edge in sorted(self.forward.get(node, []), key=lambda e: (e.relation, e.object, e.document_id)):
                next_path = path + [edge]
                if edge.relation == relation:
                    return next_path
                if edge.object not in seen:
                    seen.add(edge.object); queue.append((edge.object, next_path))
        return []

    def directed_community_edges(self, root: str, *, max_hops: int = 4) -> list[GraphEdge]:
        queue = deque([(root, 0)]); seen_nodes = {root}; seen_edges: set[tuple[str,str,str,str]] = set(); out=[]
        while queue:
            node, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for edge in self.forward.get(node, []):
                key=(edge.subject,edge.relation,edge.object,edge.document_id)
                if key not in seen_edges:
                    seen_edges.add(key); out.append(edge)
                if edge.object not in seen_nodes:
                    seen_nodes.add(edge.object); queue.append((edge.object, depth+1))
        return out

class KAGPathRetriever:
    """Structured path reasoning over explicit graph edges with document provenance."""
    def __init__(self, graph: KnowledgeGraph): self.graph=graph
    def search(self, query: str, *, k: int = 5) -> list[tuple[str,float]]:
        seeds=self.graph.match_entities(query)
        if len(seeds)>=2:
            path=self.graph.shortest_path(seeds[0], seeds[1])
        elif seeds:
            relation=self.graph.target_relation(query)
            path=self.graph.shortest_path_to_relation(seeds[0], relation, max_hops=6) if relation else []
        else:
            path=[]
        docs=[]
        for edge in path:
            if edge.document_id not in docs: docs.append(edge.document_id)
        return [(doc_id, 1.0/(i+1)) for i,doc_id in enumerate(docs[:k])]

class GlobalGraphRetriever:
    """GraphRAG-style community expansion for global/collection relation queries."""
    def __init__(self, graph: KnowledgeGraph): self.graph=graph
    def _roots(self, query: str) -> list[str]:
        matches=set(self.graph.match_entities(query)); roots=[r for r in self.graph.network_roots if r in matches]
        return roots or list(self.graph.network_roots)
    def search(self, query: str, *, k: int = 10) -> list[tuple[str,float]]:
        if k<=0:return []
        target=self.graph.target_relation(query)
        if target == "country": allowed={"includes","located_in","country"}
        elif target == "studies": allowed={"studies","includes"}
        elif target == "currency": allowed={"includes","located_in","country","currency"}
        else: allowed=None
        q=set(tokenize(query)); scores: dict[str,float]={}
        for root in self._roots(query):
            for edge in self.graph.directed_community_edges(root, max_hops=5):
                if allowed is not None and edge.relation not in allowed: continue
                rel_terms=set(tokenize(edge.relation.replace("_"," ")))
                text_terms=set(tokenize(self.graph.documents[edge.document_id].text))
                rel_bonus=3.0 if edge.relation==target else 1.0
                lexical=len(q & text_terms)*0.25 + len(q & rel_terms)*0.5
                scores[edge.document_id]=max(scores.get(edge.document_id,0.0), rel_bonus+lexical)
        ranked=sorted(scores.items(),key=lambda x:(-x[1],x[0])); return ranked[:k]

class HippoRAGRetriever:
    """Associative retrieval via query-seeded personalized PageRank over the KG."""
    def __init__(self, graph: KnowledgeGraph, *, damping: float=.85, iterations: int=24):
        self.graph=graph; self.damping=damping; self.iterations=iterations
        self.doc_index=BM25Index({d.id:d.text for d in graph.documents.values()})
    def _pagerank(self,seeds:list[str])->dict[str,float]:
        nodes=self.graph.entities
        if not nodes:return {}
        seedset=set(seeds) or set(nodes)
        teleport={n:(1/len(seedset) if n in seedset else 0.0) for n in nodes}
        score=teleport.copy()
        for _ in range(self.iterations):
            nxt={n:(1-self.damping)*teleport[n] for n in nodes}
            for node in nodes:
                neigh=self.graph.undirected.get(node,[])
                if not neigh: continue
                share=self.damping*score.get(node,0.0)/len(neigh)
                for other,_ in neigh: nxt[other]+=share
            score=nxt
        return score
    def search(self,query:str,*,k:int=5)->list[tuple[str,float]]:
        import math
        seeds=self.graph.match_entities(query)
        lexical=dict(self.doc_index.search(query,k=len(self.graph.documents)))
        maxlex=max(lexical.values(),default=1.0) or 1.0
        scores=defaultdict(float)
        if len(seeds) >= 2:
            per_seed=[self._pagerank([seed]) for seed in seeds]
            for edge in self.graph.edges:
                values=[max(pr.get(edge.subject,0.0)+pr.get(edge.object,0.0),1e-12) for pr in per_seed]
                assoc=math.prod(values)**(1.0/len(values))
                scores[edge.document_id]=max(scores[edge.document_id],assoc)
        else:
            pr=self._pagerank(seeds)
            for edge in self.graph.edges:
                assoc=pr.get(edge.subject,0.0)+pr.get(edge.object,0.0)
                scores[edge.document_id]=max(scores[edge.document_id],assoc)
        lexical_weight = 0.0 if len(seeds) >= 2 else 0.05
        for doc_id in scores:
            scores[doc_id]+=lexical_weight*lexical.get(doc_id,0.0)/maxlex
        ranked=sorted(scores.items(),key=lambda x:(-x[1],x[0])); return ranked[:k]

    def stats(self)->dict[str,int]:
        return {"graph_nodes":len(self.graph.entities),"graph_edges":len(self.graph.edges)}
