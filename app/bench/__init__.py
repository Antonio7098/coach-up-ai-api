"""Benchmarking harness core package.

Modules:
- dataset: load and validate JSONL golden set
- providers: base interface + mock provider (extensible)
- judge: heuristic judge + stubs for LLM-as-judge
- metrics: aggregation utilities
- report: Markdown report generator
"""
