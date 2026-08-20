from __future__ import annotations
from collections import defaultdict
from rag_practice.ir.bm25 import BM25Index
from .models import MemoryEvent

class TemporalMemoryIndex:
    """Version-aware episodic memory: relevance chooses a key, time chooses a version."""
    def __init__(self, events: list[MemoryEvent]):
        if not events: raise ValueError("events must not be empty")
        self.events={e.id:e for e in events}; self.by_key:dict[str,list[MemoryEvent]]=defaultdict(list)
        for event in events:self.by_key[event.memory_key].append(event)
        for values in self.by_key.values(): values.sort(key=lambda e:(e.sequence,e.id))
        self._rebuild()
    def _rebuild(self): self.index=BM25Index({e.id:e.text for e in self.events.values()})
    def add(self,event:MemoryEvent):
        if event.id in self.events: raise ValueError(f"duplicate event id: {event.id}")
        self.events[event.id]=event; self.by_key[event.memory_key].append(event); self.by_key[event.memory_key].sort(key=lambda e:(e.sequence,e.id)); self._rebuild()
    def _rank_keys(self,query:str)->list[tuple[str,float]]:
        event_scores=dict(self.index.search(query,k=len(self.events))); key_scores=defaultdict(float)
        for event_id,score in event_scores.items(): key_scores[self.events[event_id].memory_key]=max(key_scores[self.events[event_id].memory_key],score)
        return sorted(key_scores.items(),key=lambda x:(-x[1],x[0]))
    def search(self,query:str,*,k:int=3)->list[tuple[str,float]]:
        if k<=0:return []
        ranked_keys=self._rank_keys(query); previous=any(term in query.lower() for term in ("before","previous"))
        out=[]
        for key,key_score in ranked_keys:
            versions=self.by_key[key]
            chosen=versions[-2] if previous and len(versions)>=2 else versions[-1]
            out.append((chosen.id,key_score))
            if len(out)>=k: break
        return out
    def stats(self)->dict[str,int]: return {"memory_events":len(self.events),"memory_keys":len(self.by_key)}
