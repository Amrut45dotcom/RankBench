"""Checks the LLM's answer in code instead of trusting the prompt.

What it checks: format and citation existence only.
  - answer is either exactly INSUFFICIENT_CONTEXT, or a real answer
  - every cited passage number exists in the passages that were sent
  - every sentence carries at least one citation
What it does NOT check: whether the cited passage really supports the claim.
That is the faithfulness check, done by hand on ~40 answers.
"""
import re
import unicodedata

REFUSAL_TOKEN = "INSUFFICIENT_CONTEXT"

# bracket look-alikes the model sometimes emits, mapped to plain [ ]
_BRACKETS = {"【": "[", "】": "]", "［": "[", "］": "]", "〔": "[", "〕": "]"}
# dash look-alikes (non-breaking hyphen etc.) mapped to a plain hyphen
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")

_CITE_GROUP = r"\[\s*P\d+(?:\s*,\s*P\d+)*\s*\]"
_CITE = re.compile(_CITE_GROUP)
_CITES_AFTER_PUNCT = re.compile(r"([.!?]+)\s*((?:" + _CITE_GROUP + r"\s*)+)")


def clean_citations(text):
    """Fix bracket look-alikes only. Safe to show to the user."""
    for bad, good in _BRACKETS.items():
        text = text.replace(bad, good)
    return text


def normalize(text):
    """Aggressive normalization, used only for checking, never shown to the user."""
    text = unicodedata.normalize("NFKC", text)
    text = clean_citations(text)
    text = text.translate(_DASHES)
    return text.strip()


def _move_cites_before_punct(text):
    """'claim. [P1] Next claim.' -> 'claim [P1]. Next claim.' so each citation
    stays with the sentence it follows."""
    return _CITES_AFTER_PUNCT.sub(lambda m: " " + m.group(2).strip() + m.group(1) + " ", text)


def _split_sentences(text):
    pieces = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            pieces.extend(re.split(r"(?<=[.!?])\s+", line))
    sentences = []
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        leftover = _CITE.sub("", p).strip(" .!?,;:*_-")
        if not leftover and sentences:      # citation-only fragment: attach to previous sentence
            sentences[-1] += " " + p
        else:
            sentences.append(p)
    return sentences


def cited_numbers(text):
    """Ordered unique passage numbers cited in the text, e.g. [P1][P3] -> [1, 3]."""
    seen = []
    for group in _CITE.findall(text):
        for n in re.findall(r"P(\d+)", group):
            if int(n) not in seen:
                seen.append(int(n))
    return seen


def validate(raw, n_passages):
    """Returns {"status": "ok" | "refused" | "invalid", "reasons": [...], "cited": [...]}."""
    text = normalize(raw or "")
    if not text:
        return {"status": "invalid", "reasons": ["empty_answer"], "cited": []}

    if REFUSAL_TOKEN in text:
        if text.strip(" .*_`") == REFUSAL_TOKEN:
            return {"status": "refused", "reasons": [], "cited": []}
        return {"status": "invalid", "reasons": ["refusal_mixed_with_answer"], "cited": []}

    reasons = []
    cited = cited_numbers(text)
    if not cited:
        reasons.append("no_citations")
    unknown = [n for n in cited if n < 1 or n > n_passages]
    if unknown:
        reasons.append(f"unknown_citation:{unknown}")
    if not _CITE.sub("", text).strip(" .!?,;:*_-"):
        reasons.append("no_content")

    sentences = _split_sentences(_move_cites_before_punct(text))
    uncited = [s for s in sentences if not _CITE.search(s)]
    if uncited:
        reasons.append(f"uncited_sentences:{len(uncited)}")

    return {"status": "invalid" if reasons else "ok", "reasons": reasons, "cited": cited}
