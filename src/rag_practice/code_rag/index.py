"""Low-level Python repository indexing for the Code RAG sub-lab."""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rag_practice.ir.bm25 import BM25Index
from rag_practice.sources.base import SourceHit, SourceRecord


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class CodeSymbol:
    """One Python definition with an exact repository source span."""

    id: str
    path: str
    qualname: str
    name: str
    kind: str
    line: int
    end_line: int
    source: str
    docstring: str
    raw_calls: tuple[str, ...]

    @property
    def locator(self) -> str:
        return f"code://repo/{self.path}#L{self.line}-L{self.end_line}"


class PythonRepositoryIndex:
    """AST-derived file/symbol index plus a small static call graph.

    The implementation deliberately avoids Tree-sitter or orchestration frameworks so
    symbol boundaries, call resolution, scoring, and graph expansion remain inspectable.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.files: dict[str, str] = {}
        self.symbols: dict[str, CodeSymbol] = {}
        self.imports: dict[str, dict[str, str]] = {}
        self.file_modules: dict[str, str] = {}

        for path in sorted(self.root.rglob("*.py")):
            relative = path.relative_to(self.root).as_posix()
            text = path.read_text()
            tree = ast.parse(text)
            self.files[relative] = text
            self.file_modules[relative] = ".".join(Path(relative).with_suffix("").parts)
            self.imports[relative] = self._imports_for(tree)

            lines = text.splitlines()
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self._add_symbol(relative, node, node.name, "function", lines)
                elif isinstance(node, ast.ClassDef):
                    self._add_symbol(relative, node, node.name, "class", lines)
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            self._add_symbol(
                                relative,
                                child,
                                f"{node.name}.{child.name}",
                                "method",
                                lines,
                            )

        self.by_name: dict[str, list[str]] = defaultdict(list)
        self.by_dotted: dict[str, str] = {}
        for symbol in self.symbols.values():
            self.by_name[symbol.name].append(symbol.id)
            module = self.file_modules[symbol.path]
            self.by_dotted[f"{module}.{symbol.qualname}"] = symbol.id
            self.by_dotted[f"{module}.{symbol.name}"] = symbol.id

        self.call_graph: dict[str, list[str]] = {symbol_id: [] for symbol_id in self.symbols}
        self.reverse_call_graph: dict[str, list[str]] = {
            symbol_id: [] for symbol_id in self.symbols
        }
        for symbol in self.symbols.values():
            for raw_call in symbol.raw_calls:
                target = self._resolve_call(symbol.path, raw_call)
                if target and target not in self.call_graph[symbol.id]:
                    self.call_graph[symbol.id].append(target)
                    self.reverse_call_graph[target].append(symbol.id)

        self._file_bm25 = BM25Index(self.files)
        self._symbol_documents = {
            symbol.id: " ".join(
                [
                    symbol.path,
                    symbol.qualname,
                    symbol.name,
                    symbol.kind,
                    symbol.docstring,
                    symbol.source,
                ]
            )
            for symbol in self.symbols.values()
        }
        self._symbol_bm25 = BM25Index(self._symbol_documents)

    @staticmethod
    def _imports_for(tree: ast.Module) -> dict[str, str]:
        imports: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.asname or alias.name.split(".")[0]] = alias.name
        return imports

    def _add_symbol(
        self,
        path: str,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        qualname: str,
        kind: str,
        lines: list[str],
    ) -> None:
        source = "\n".join(lines[node.lineno - 1 : node.end_lineno])
        raw_calls: list[str] = []
        if kind != "class":
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                function = child.func
                if isinstance(function, ast.Name):
                    raw_calls.append(function.id)
                elif isinstance(function, ast.Attribute):
                    if isinstance(function.value, ast.Name):
                        raw_calls.append(f"{function.value.id}.{function.attr}")
                    else:
                        raw_calls.append(function.attr)

        name = qualname.split(".")[-1]
        symbol_id = f"{path}::{qualname}"
        self.symbols[symbol_id] = CodeSymbol(
            id=symbol_id,
            path=path,
            qualname=qualname,
            name=name,
            kind=kind,
            line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            source=source,
            docstring=ast.get_docstring(node) or "",
            raw_calls=tuple(raw_calls),
        )

    def _resolve_call(self, path: str, raw_call: str) -> str | None:
        """Resolve only call forms that are unambiguous in this teaching index.

        Attribute calls on ordinary local variables are deliberately left unresolved.
        Treating ``rates.get`` as the unique method named ``get`` elsewhere in a
        repository would be a classic false graph edge.
        """

        imports = self.imports[path]
        base = raw_call.split(".")[-1]

        if "." in raw_call:
            prefix, suffix = raw_call.split(".", 1)
            if prefix not in imports:
                return None
            return self.by_dotted.get(f"{imports[prefix]}.{suffix}")

        if base in imports and imports[base] in self.by_dotted:
            return self.by_dotted[imports[base]]

        candidates = self.by_name.get(base, [])
        if len(candidates) == 1:
            return candidates[0]
        return None

    @staticmethod
    def _requested_symbol(query: str) -> str | None:
        patterns = [
            r"(?:where is|implementation of|find the implementation of)\s+"
            r"([A-Za-z_][A-Za-z0-9_]*)",
            r"([A-Za-z_][A-Za-z0-9_]*)\s+(?:implemented|defined)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def search_files(self, query: str, *, k: int = 4) -> list[tuple[str, float]]:
        return self._file_bm25.search(query, k=k)

    def search_symbols(self, query: str, *, k: int = 4) -> list[tuple[str, float]]:
        return self._symbol_bm25.search(query, k=k)

    def search_symbol_graph(self, query: str, *, k: int = 4) -> list[tuple[str, float]]:
        """Fuse lexical symbol ranking with explicit-name and call-graph evidence."""

        lexical = self._symbol_bm25.search(query, k=max(k, 12))
        lexical_scores = dict(lexical)
        scores = dict(lexical_scores)
        identifiers = set(_IDENTIFIER_RE.findall(query))
        requested = self._requested_symbol(query)

        for symbol in self.symbols.values():
            if symbol.name == requested:
                scores[symbol.id] = scores.get(symbol.id, 0.0) + 25.0
            elif symbol.name in identifiers:
                scores[symbol.id] = scores.get(symbol.id, 0.0) + 5.0

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        lowered = query.lower()
        reverse_mode = any(word in lowered for word in ("caller", "rename", "renamed"))
        outgoing_mode = any(
            phrase in lowered
            for phrase in (
                " call ",
                " calls ",
                "before returning",
                "including",
                "helper",
                "rename",
                "renamed",
            )
        )

        if reverse_mode or outgoing_mode:
            exact_seeds = [
                symbol_id
                for symbol_id, _ in ranked
                if self.symbols[symbol_id].name in identifiers
            ]
            seeds = (exact_seeds or [symbol_id for symbol_id, _ in ranked])[:2]
            promote_neighbors = (
                reverse_mode
                or ("does " in lowered and "call" in lowered)
                or "which functions does" in lowered
                or "which helper" in lowered
            )

            for seed in seeds:
                neighbors = (
                    self.reverse_call_graph[seed]
                    if reverse_mode
                    else self.call_graph[seed]
                )
                seed_score = scores.get(seed, 0.0)
                for neighbor in neighbors:
                    offset = 5.0 if promote_neighbors else -0.2
                    expanded_score = (
                        seed_score
                        + offset
                        + 0.1 * lexical_scores.get(neighbor, 0.0)
                    )
                    scores[neighbor] = max(scores.get(neighbor, 0.0), expanded_score)

            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

        return ranked[:k]

    def symbol_record(self, symbol_id: str) -> SourceRecord:
        symbol = self.symbols[symbol_id]
        return SourceRecord(
            id=symbol.id,
            source_type="code",
            locator=symbol.locator,
            title=symbol.qualname,
            content=symbol.source,
            metadata={
                "path": symbol.path,
                "qualname": symbol.qualname,
                "name": symbol.name,
                "kind": symbol.kind,
                "line": symbol.line,
                "end_line": symbol.end_line,
                "calls": tuple(self.call_graph[symbol.id]),
            },
        )

    def search(self, query: str, *, limit: int = 5) -> list[SourceHit]:
        """Expose the symbol-aware path through the shared M08 Source contract."""

        return [
            SourceHit(
                record=self.symbol_record(symbol_id),
                score=score,
                rank=rank,
                details={"retrieval": "symbol_graph"},
            )
            for rank, (symbol_id, score) in enumerate(
                self.search_symbol_graph(query, k=limit), start=1
            )
        ]
