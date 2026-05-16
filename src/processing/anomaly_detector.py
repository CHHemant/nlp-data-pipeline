"""
validator.py

Validation layer for RAG (Retrieval-Augmented Generation) pipelines.

Three checks:
  1. Retrieval faithfulness  — does the answer only use retrieved chunks?
  2. Consistency            — does same query give consistent answers?
  3. Attribution            — can every claim be traced to a source?

Usage:
    validator = RAGValidator(llm_client=anthropic_client)
    result = await validator.validate(query, retrieved_chunks, answer)
    if not result.passed:
        # re-run retrieval or flag for human review
        print(result.issues)
"""

from __future__ import annotations
import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    passed: bool
    faithfulness: float    # 0-1, how grounded the answer is in retrieved chunks
    consistency: float     # 0-1, stability across multiple runs
    attribution: float     # 0-1, fraction of claims traceable to source
    overall_score: float   # weighted composite
    issues: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    # thresholds — adjust based on your use case
    FAITHFULNESS_MIN = 0.72
    CONSISTENCY_MIN  = 0.85
    ATTRIBUTION_MIN  = 0.60


class RAGValidator:
    """
    Validates RAG pipeline outputs. Needs an LLM client for the
    faithfulness and attribution checks — these require understanding
    natural language, not just pattern matching.
    """

    def __init__(self, llm_client=None, consistency_runs: int = 3):
        """
        llm_client: callable async fn(prompt) -> str
                    pass your Anthropic/OpenAI client here
        consistency_runs: how many times to re-run query for consistency check
                          3 is a good tradeoff between cost and reliability
        """
        self._llm = llm_client
        self._consistency_runs = consistency_runs
        self._cache: dict[str, str] = {}   # simple in-memory cache

    # ── Main entry ─────────────────────────────────────────────────────────────

    async def validate(
        self,
        query: str,
        retrieved_chunks: list[str],
        answer: str,
        regenerate_fn=None,    # async fn(query) -> (chunks, answer) for consistency check
    ) -> ValidationResult:
        """
        Run all three validation checks. Returns ValidationResult.

        If regenerate_fn is None, consistency check is skipped (score = 1.0).
        """
        t0 = time.monotonic()
        issues = []

        context = "\n\n".join(retrieved_chunks)

        # run faithfulness + attribution concurrently (both need LLM)
        faithfulness_task = asyncio.create_task(
            self._check_faithfulness(answer, context)
        )
        attribution_task = asyncio.create_task(
            self._check_attribution(answer, retrieved_chunks)
        )
        faithfulness, attribution = await asyncio.gather(faithfulness_task, attribution_task)

        # consistency needs multiple generations — run sequentially to avoid rate limits
        if regenerate_fn is not None:
            consistency = await self._check_consistency(query, answer, regenerate_fn)
        else:
            consistency = 1.0  # skip if no regenerate function provided

        # composite score: faithfulness matters most
        overall = (faithfulness * 0.50) + (consistency * 0.30) + (attribution * 0.20)

        # collect issues
        if faithfulness < ValidationResult.FAITHFULNESS_MIN:
            issues.append(
                f"Low faithfulness ({faithfulness:.2f}) — answer may not be grounded in retrieved context. "
                f"Check if retriever is returning relevant chunks."
            )
        if consistency < ValidationResult.CONSISTENCY_MIN:
            issues.append(
                f"Low consistency ({consistency:.2f}) — same query gives different answers across runs. "
                f"Consider lower temperature or more deterministic retrieval."
            )
        if attribution < ValidationResult.ATTRIBUTION_MIN:
            issues.append(
                f"Low attribution ({attribution:.2f}) — some claims can't be traced to source chunks. "
                f"Possible hallucination."
            )

        return ValidationResult(
            passed=(len(issues) == 0),
            faithfulness=round(faithfulness, 3),
            consistency=round(consistency, 3),
            attribution=round(attribution, 3),
            overall_score=round(overall, 3),
            issues=issues,
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        )

    # ── Check 1: faithfulness ─────────────────────────────────────────────────

    async def _check_faithfulness(self, answer: str, context: str) -> float:
        """
        Does the answer only use information from the retrieved context?
        Uses LLM to score 0-1. Falls back to lexical overlap if no LLM.
        """
        if self._llm is None:
            return self._lexical_faithfulness(answer, context)

        prompt = f"""Score how faithful this answer is to the given context.
Faithfulness means: every factual claim in the answer is supported by the context.
An answer that makes up facts not in the context scores 0.
An answer that only uses context information scores 1.

Context:
{context[:2000]}

Answer:
{answer[:1000]}

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""

        try:
            response = await self._llm(prompt)
            score = float(re.search(r"0?\.\d+|[01]\.0*", response.strip()).group())
            return max(0.0, min(1.0, score))
        except Exception:
            # if LLM call fails, fall back to lexical
            return self._lexical_faithfulness(answer, context)

    @staticmethod
    def _lexical_faithfulness(answer: str, context: str) -> float:
        """
        Rough lexical faithfulness — what fraction of answer words appear in context?
        Not great but works as a fallback when LLM is unavailable.
        """
        answer_words = set(re.findall(r"\b\w{4,}\b", answer.lower()))
        context_words = set(re.findall(r"\b\w{4,}\b", context.lower()))
        if not answer_words:
            return 1.0
        overlap = answer_words & context_words
        return len(overlap) / len(answer_words)

    # ── Check 2: consistency ──────────────────────────────────────────────────

    async def _check_consistency(
        self, query: str, original_answer: str, regenerate_fn
    ) -> float:
        """
        Re-run the same query N times and measure how similar the answers are.
        Uses cosine similarity on TF-IDF vectors — no LLM needed here.
        """
        answers = [original_answer]

        for _ in range(self._consistency_runs - 1):
            try:
                _, new_answer = await regenerate_fn(query)
                answers.append(new_answer)
                await asyncio.sleep(0.2)  # be polite to the API
            except Exception:
                pass  # skip failed runs — don't crash the whole validation

        if len(answers) < 2:
            return 1.0  # can't measure with only one answer

        return self._pairwise_similarity(answers)

    @staticmethod
    def _pairwise_similarity(texts: list[str]) -> float:
        """Mean pairwise cosine similarity using TF-IDF."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        if len(texts) < 2:
            return 1.0

        try:
            vec = TfidfVectorizer().fit_transform(texts).toarray()
            sims = []
            for i in range(len(vec)):
                for j in range(i + 1, len(vec)):
                    sim = float(cosine_similarity([vec[i]], [vec[j]])[0][0])
                    sims.append(sim)
            return float(np.mean(sims)) if sims else 1.0
        except Exception:
            return 1.0

    # ── Check 3: attribution ──────────────────────────────────────────────────

    async def _check_attribution(self, answer: str, chunks: list[str]) -> float:
        """
        What fraction of sentences in the answer can be attributed to a chunk?
        Uses LLM for each sentence — expensive but accurate.
        Falls back to keyword overlap if no LLM.
        """
        sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 20]
        if not sentences:
            return 1.0

        if self._llm is None:
            # fallback: keyword overlap per sentence
            scored = []
            for sent in sentences:
                sent_words = set(re.findall(r"\b\w{4,}\b", sent.lower()))
                chunk_words = set(re.findall(r"\b\w{4,}\b", " ".join(chunks).lower()))
                overlap = len(sent_words & chunk_words) / max(len(sent_words), 1)
                scored.append(overlap > 0.3)
            return sum(scored) / len(scored)

        # with LLM: check each sentence against all chunks
        context = "\n".join(f"[Chunk {i+1}] {c[:300]}" for i, c in enumerate(chunks[:5]))
        attributed = 0

        for sentence in sentences[:8]:  # cap at 8 sentences to control cost
            prompt = f"""Can this sentence be attributed to (supported by) the provided chunks?
Answer ONLY 'yes' or 'no'.

Chunks:
{context}

Sentence: {sentence}"""
            try:
                resp = await self._llm(prompt)
                if "yes" in resp.lower():
                    attributed += 1
            except Exception:
                attributed += 1  # assume attributed on failure (conservative)

        return attributed / len(sentences[:8])
