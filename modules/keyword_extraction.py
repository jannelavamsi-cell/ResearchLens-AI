"""
Keyword and technical concept extraction engine for ResearchLens AI.
Combines KeyBERT (with MMR diversity), TF-IDF across the collection,
and spaCy noun chunk / technical terminology extraction.
"""

from typing import Any, Dict, List, Set, Tuple

import spacy
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer

from core.logging_config import get_logger

logger = get_logger("keyword_extraction")

# Domain stopwords that frequently dominate academic text but carry minimal conceptual specificity
ACADEMIC_DOMAIN_STOPWORDS: Set[str] = {
    "paper", "approach", "model", "models", "results", "method", "methods",
    "dataset", "datasets", "proposed", "work", "author", "authors", "study",
    "table", "figure", "section", "et", "al", "using", "based", "evaluation",
    "experiments", "experimental", "shown", "shows", "performance", "state",
    "art", "different", "number", "task", "tasks", "findings", "analysis",
    "data", "technique", "techniques", "system", "systems", "result", "compare",
    "baseline", "baselines", "present", "presents", "presented"
}

_nlp = None
_keybert_model = None


def get_spacy_nlp():
    """Lazy loader for spaCy pipeline."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            _nlp = spacy.blank("en")
    return _nlp


def get_keybert():
    """Lazy loader for KeyBERT model."""
    global _keybert_model
    if _keybert_model is None:
        _keybert_model = KeyBERT(model="all-MiniLM-L6-v2")
    return _keybert_model


class KeywordExtractor:
    """Extracts, merges, and scores key concepts across academic papers."""

    def __init__(self, top_n: int = 15) -> None:
        self.top_n = top_n

    def extract_spacy_terms(self, text: str, max_terms: int = 20) -> List[Tuple[str, float]]:
        """Extracts noun chunks and technical entities filtered by length and academic stoplist."""
        nlp = get_spacy_nlp()
        doc = nlp(text[:10000])  # Sample first 10,000 characters for speed

        candidates = []
        seen = set()

        # Noun chunks
        if hasattr(doc, "noun_chunks"):
            for chunk in doc.noun_chunks:
                clean = chunk.text.lower().strip()
                # Exclude stopwords, single letters, and long strings
                if 3 <= len(clean) <= 40 and len(clean.split()) <= 3 and clean not in seen:
                    words = set(clean.split())
                    # Skip if all words in chunk are generic academic stopwords
                    if words.issubset(ACADEMIC_DOMAIN_STOPWORDS):
                        continue
                    seen.add(clean)
                    candidates.append((clean, 0.5))

        return candidates[:max_terms]

    def extract_keybert_terms(self, text: str, max_terms: int = 15) -> List[Tuple[str, float]]:
        """Extracts diverse keyphrases across unigrams, bigrams, and trigrams using KeyBERT (MMR)."""
        try:
            kb = get_keybert()
            # Use MMR for diversity across n-grams (1 to 3)
            raw_keywords = kb.extract_keywords(
                text[:15000],
                keyphrase_ngram_range=(1, 3),
                stop_words="english",
                use_mmr=True,
                diversity=0.7,
                top_n=max_terms * 2,
            )
            filtered = []
            for phrase, score in raw_keywords:
                clean_phrase = phrase.lower().strip()
                words = set(clean_phrase.split())
                if not words.issubset(ACADEMIC_DOMAIN_STOPWORDS) and len(clean_phrase) >= 3:
                    filtered.append((clean_phrase, score))
                if len(filtered) >= max_terms:
                    break
            return filtered
        except Exception as e:
            logger.warning(f"KeyBERT extraction fallback: {e}")
            return []

    def extract_tfidf_terms(
        self, target_text: str, corpus_texts: List[str], max_terms: int = 15
    ) -> List[Tuple[str, float]]:
        """Calculates TF-IDF terms for target_text relative to collection corpus."""
        all_docs = [target_text] + [c for c in corpus_texts if c != target_text]
        if len(all_docs) < 1:
            return []

        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

            combined_stopwords = list(ENGLISH_STOP_WORDS.union(ACADEMIC_DOMAIN_STOPWORDS))
            vec = TfidfVectorizer(
                stop_words=combined_stopwords,
                ngram_range=(1, 3),
                max_features=250,
            )
            tfidf_mat = vec.fit_transform(all_docs)
            feature_names = vec.get_feature_names_out()

            # Target doc is index 0
            scores = tfidf_mat[0].toarray()[0]
            top_indices = scores.argsort()[::-1][:max_terms * 2]

            results = []
            for idx in top_indices:
                if scores[idx] > 0.0:
                    term = feature_names[idx]
                    if len(term) >= 3 and not set(term.split()).issubset(ACADEMIC_DOMAIN_STOPWORDS):
                        results.append((term, float(scores[idx])))
                if len(results) >= max_terms:
                    break
            return results
        except Exception as e:
            logger.debug(f"TF-IDF extraction error: {e}")
            return []

    def extract_paper_keywords(
        self, paper_text: str, collection_corpus: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Combines KeyBERT, TF-IDF, and spaCy noun phrases.
        Merges duplicates by max score and returns sorted top_n keywords.
        """
        combined_scores: Dict[str, float] = {}
        term_methods: Dict[str, str] = {}

        # 1. KeyBERT
        kb_terms = self.extract_keybert_terms(paper_text, max_terms=self.top_n)
        for term, score in kb_terms:
            t = term.lower().strip()
            combined_scores[t] = max(combined_scores.get(t, 0.0), float(score))
            term_methods[t] = "keybert"

        # 2. TF-IDF across collection
        if collection_corpus:
            tfidf_terms = self.extract_tfidf_terms(paper_text, collection_corpus, max_terms=self.top_n)
            for term, score in tfidf_terms:
                t = term.lower().strip()
                combined_scores[t] = max(combined_scores.get(t, 0.0), float(score))
                if t not in term_methods:
                    term_methods[t] = "tfidf"

        # 3. spaCy noun chunks fallback
        spacy_terms = self.extract_spacy_terms(paper_text, max_terms=10)
        for term, score in spacy_terms:
            t = term.lower().strip()
            if t not in combined_scores:
                combined_scores[t] = score
                term_methods[t] = "spacy"

        # Sort and take top_n
        sorted_terms = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[: self.top_n]

        results = [
            {
                "keyword": term,
                "score": round(score, 4),
                "method": term_methods.get(term, "merged"),
            }
            for term, score in sorted_terms
        ]

        logger.info(f"Extracted {len(results)} merged keywords.")
        return results
