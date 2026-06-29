import re
import unicodedata
from dataclasses import dataclass, field


_ZERO_WIDTH_RE = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")


# Unambiguously W3C-domain terms — matching any of these means high-confidence in-scope
STRONG_TOPIC_KEYWORDS = frozenset([
    "fpwd", "working draft", "candidate recommendation", "cr", "crd", "crs",
    "recommendation", "rec", "proposed recommendation", "推荐标准", "候选推荐",
    "charter", "recharter", "章程",
    "staff contact", "team contact", "职责",
    "horizontal review", "wide review", "横向审查",
    "formal objection", "appeal", "异议", "申诉",
    "patent", "ipr", "专利", "pubrules",
    "ac review", "transition", "转换",
    "working group", "interest group", "community group", "工作组",
    "working draft", "specification", "标准",
    "workshop", "workshops", "tpac", "研讨会",
])

PROCESS_TOPICS = {
    "process": ["process", "流程", "程序", "process document"],
    "recommendation_track": [
        "fpwd",
        "working draft",
        "wd",
        "candidate recommendation",
        "cr",
        "crd",
        "crs",
        "recommendation",
        "rec",
        "spec",
        "specification",
        "proposed recommendation",
        "标准",
        "推荐标准",
        "候选推荐",
    ],
    "governance": [
        "charter",
        "working group",
        "interest group",
        "community group",
        "staff contact",
        "team contact",
        "chair",
        "chairs",
        "meeting",
        "meetings",
        "liaison",
        "wg",
        "ig",
        "cg",
        "章程",
        "工作组",
        "主席",
        "会议",
        "联络",
        "职责",
    ],
    "review": ["wide review", "horizontal review", "ac review", "transition", "review", "审查", "转换"],
    "objection": ["formal objection", "appeal", "异议", "申诉"],
    "policy": ["patent", "ipr", "pubrules", "code of conduct", "专利", "发布规则"],
    "guidebook": ["guidebook", "guide", "art of consensus", "指南"],
    "w3c": ["w3c", "万维网联盟"],
    "events": ["workshop", "workshops", "tpac", "ac meeting", "advisory committee meeting", "breakout", "研讨会"],
    # Publication / Communications workflow. Without these, questions
    # like "how to announce new publications?" get rejected by the
    # keyword scope gate in template / no-router mode — the LLM
    # router rescues them in production but the deterministic eval
    # has no router fallback. These ARE legitimate Process topics
    # (Process §7.1 covers Publication and Communication).
    "communications": [
        "publication",
        "publish",
        "publishing",
        "announce",
        "announcement",
        "announcing",
        "press release",
        "communications team",
        "w3t-comm",
        "www-announce",
        "call for review",
        "cfr",
        "公告",
        "宣布",
        "发布",
    ],
}

# Patterns where the user mentions W3C but is clearly NOT asking about
# Process workflow. Weak ``w3c`` keyword matches alone shouldn't make these
# in-scope; this list catches the obvious cases (jokes, trivia, analogies)
# without depending on the LLM router being available.
FRIVOLOUS_PATTERNS = (
    "joke about",
    "tell me a joke",
    "knock knock",
    "笑话",
    "幽默",
    "cooking recipe",
    "cooking process recipe",
    "as a cooking",
    "as a recipe",
    "explain like i'm 5",
    "explain like i am 5",
    "eli5",
    "in the style of",
    "in style of",
    "when was the w3c founded",
    "when was w3c founded",
    "history of w3c",
    "history of the w3c",
    "who founded w3c",
    "who founded the w3c",
    "w3c trivia",
)


def detect_frivolous(text: str) -> bool:
    """True when the message looks like W3C-name-dropping without a real
    Process question. Used to override weak keyword matches that would
    otherwise let "Tell me a joke about W3C" or "When was W3C founded?"
    leak past the scope gate.
    """
    normalized = _normalize_for_injection_scan(text)
    return any(pattern in normalized for pattern in FRIVOLOUS_PATTERNS)


INJECTION_PATTERNS = [
    "ignore previous",
    "ignore your",
    "system prompt",
    "hidden prompt",
    "developer message",
    "do not cite",
    "不要引用",
    "忽略之前",
    "系统提示词",
    "隐藏提示",
    "这是新版 process",
    "this is the new process",
]


@dataclass(frozen=True)
class ScopeDecision:
    in_scope: bool
    reason: str
    matched_topics: list[str]
    injection_risk: bool
    # 0.9 = strong W3C-specific keyword matched
    # 0.5 = only generic terms like "w3c"/"process"/"guide" matched (weak)
    # 0.0 = no match
    confidence: float = 1.0


def _normalize_for_injection_scan(text: str) -> str:
    """Defeat trivial bypasses: Unicode homoglyphs, zero-width characters, case."""
    # NFKC folds compatibility characters (full-width, ligatures) and many
    # homoglyph variants into their canonical ASCII forms.
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _ZERO_WIDTH_RE.sub("", normalized)
    return normalized.lower()


def detect_injection(text: str) -> bool:
    """Apply normalization before pattern matching so trivial obfuscations fail."""
    normalized = _normalize_for_injection_scan(text)
    return any(pattern in normalized for pattern in INJECTION_PATTERNS)


def classify_scope(message: str, *, history_text: str = "") -> ScopeDecision:
    text = message.lower()
    matched = [
        topic
        for topic, keywords in PROCESS_TOPICS.items()
        if any(keyword.lower() in text for keyword in keywords)
    ]
    # Scan the current message AND any provided conversation history.
    # An attacker can spread an injection across history turns.
    injection_risk = detect_injection(message) or (
        bool(history_text) and detect_injection(history_text)
    )

    if not matched:
        return ScopeDecision(False, "Question is outside the W3C Process assistant scope.", [], injection_risk, confidence=0.0)

    has_strong_match = any(kw in text for kw in STRONG_TOPIC_KEYWORDS)

    # Frivolous override: questions that mention W3C but are clearly trivia,
    # humour, or analogy ("joke about w3c", "history of W3C", "explain as
    # a cooking recipe") should be rejected even if a keyword matched.
    # Apply this for BOTH weak and strong matches because terms like "rec"
    # substring-match "recipe" and "process" matches "cooking process",
    # which would otherwise force a strong-confidence in-scope verdict.
    if detect_frivolous(message):
        return ScopeDecision(
            in_scope=False,
            reason="Question mentions W3C but is trivia / humour / analogy, not a Process workflow question.",
            matched_topics=matched,
            injection_risk=injection_risk,
            confidence=0.0,
        )

    confidence = 0.9 if has_strong_match else 0.5
    return ScopeDecision(True, "Question matches W3C Process, Guidebook, or standards workflow topics.", matched, injection_risk, confidence=confidence)
