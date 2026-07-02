import re
import unicodedata
from dataclasses import dataclass, field


_ZERO_WIDTH_RE = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")


# Unambiguously W3C-domain terms — matching any of these means high-confidence in-scope
# Round 35: bare 2-letter ``cr`` / ``rec`` / ``wd`` REMOVED. They
# substring-match harmless words ("re-CR-eation", "REC-ipe",
# "di-REC-tor", "REC-ommend", "a-WD-justment"), letting non-W3C
# queries like "recommend a good Italian restaurant" pass the
# scope gate. Real W3C questions about CR/REC/WD use the longer
# forms ("Candidate Recommendation" / "Working Draft") or
# preposition-bounded phrases ("to CR" / "from REC") that the
# task planner's ``has_stage_term`` check handles separately.
STRONG_TOPIC_KEYWORDS = frozenset([
    "fpwd", "working draft", "candidate recommendation", "crd", "crs",
    "recommendation", "proposed recommendation", "推荐标准", "候选推荐",
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
        # Bare 2-letter ``wd`` / ``cr`` / ``rec`` REMOVED — see
        # STRONG_TOPIC_KEYWORDS comment above. Longer forms below
        # cover real W3C usage without false positives.
        # Preposition-bounded stage shorthand catches the natural
        # "from CR" / "to REC" / "CR to PR" phrasings without
        # leaking on "diREctor" or "recommend a restaurant".
        " cr ", " cr,", "cr to", "from cr", "to cr", "in cr", "at cr",
        " rec ", " rec,", "rec to", "from rec", "to rec", "in rec",
        " pr ", " pr,", "pr to", "from pr", "to pr",
        " wd ", " wd,", "wd to", "from wd", "to wd",
        "candidate recommendation",
        "crd",
        "crs",
        "recommendation",
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
        # Advisory Committee + Advisory Board roles. Without these,
        # "who is the AC representative for my org?" got rejected by
        # the scope gate even though AC reps are a core Process role.
        "advisory committee",
        "advisory committee representative",
        "ac representative",
        "ac rep",
        "ac reps",
        "advisory board",
        "ab member",
        "tag member",
        "elected body",
        "elections",
        "election",
        "nominations",
        # Editor as a Process role (add/remove editor questions).
        "editor",
        "spec editor",
        "deliverable editor",
        # Scribe is a Process role (meeting minutes). The
        # task_planner already routes scribe questions to
        # run_group_process, but the SCOPE gate also needs to
        # recognise the bare "how to scribe?" phrasing.
        "scribe",
        "scribing",
        "minute-taker",
        # Invited Expert is a Process role for non-Member individuals
        # contributing to a WG. Missing before round 35 meant
        # "does Invited Expert participation have a fee?" was
        # falsely rejected.
        "invited expert",
        "invited experts",
        # Specific mailing-list types — questions about which lists
        # are public vs member-only, or how AC lists work.
        "ac mailing list",
        "ac list",
        "public-",
        "member-",
        "member-only document",
        "member-only resource",
        # Group-lifecycle events. Round 34 audit found these were
        # all rejected as OUT-OF-SCOPE because the governance topic
        # didn't include them as keywords.
        "nominate",
        "nomination",
        "close a working group",
        "close a wg",
        "close the working group",
        "close the group",
        "group closure",
        "group closes",
        "wg closes",
        "the group closes",
        "wind down",
        "terminate",
        "rescind",
        "obsolete",
        "superseded",
        "suspend",
        "suspension",
        "participant suspension",
        # CR exit criteria — "implementation" alone is too broad;
        # tie to W3C context.
        "independent implementation",
        "two implementations",
        "implementation report",
        "implementation experience",
        "implementation evidence",
        "adequate implementation",
        # Sub-group structures inside a WG — Task Forces, plus the
        # WG/IG/CG/BG taxonomy. Without these, comparison questions
        # ("what's a Business Group vs a Community Group?", "what is
        # a Task Force?") could fall through unless "Working Group"
        # was also mentioned.
        "business group",
        " bg ",
        "task force",
        "task forces",
        # Group decision mechanics — consensus + voting are core
        # Process §3.4 topics. Use phrasal keywords (not bare
        # ``consensus``) so unrelated "editor consensus" or
        # "consensus estimate" phrases don't force scope in.
        "wg vote",
        "wg voting",
        "voting in a wg",
        "voting in a working group",
        "working group vote",
        "working group voting",
        "art of consensus",
        # "Director" historically + present rare-role question
        # ("does W3C still have a Director?"). The Director role
        # was retired in the 2023 Process; questions about it
        # remain legitimate Process governance.
        "director",
        "former director",
        "章程",
        "工作组",
        "主席",
        "会议",
        "联络",
        "职责",
    ],
    "review": ["wide review", "horizontal review", "ac review", "transition", "review", "审查", "转换"],
    "objection": ["formal objection", "appeal", "异议", "申诉"],
    "policy": [
        "patent",
        "ipr",
        "pubrules",
        "code of conduct",
        "专利",
        "发布规则",
        # Normative references to external (non-W3C) standards are
        # a Process concern — REC requires that external normative
        # references be appropriately stable. These keywords let
        # questions like "how to reference an IETF RFC as a
        # normative reference?" pass the scope gate.
        "normative reference",
        "normative references",
        "external standard",
        "external normative",
    ],
    # External-org coordination — WHATWG / IETF / ISO. Without these,
    # liaison questions like "how to liaise with IETF on a normref?"
    # got falsely rejected because none of the substantive keywords
    # were in scope (only "process" / "w3c" would catch them, but the
    # question often omits both).
    "external_orgs": [
        "whatwg",
        "ietf",
        "rfc",
        " iso ",
        "iso standard",
        "ecma",
        "ecmascript",
        "ieee",
        "living standard",
        "living standards",
        "html living standard",
    ],
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
    # Round 35: defense-in-depth against the ``rec`` /
    # ``spec`` substring problem. Even after pruning bare 2-letter
    # keywords, harmless phrases like "recommend a restaurant" or
    # "specific to my city" could still pass scope via other
    # incidental matches. These literal phrases short-circuit the
    # scope gate to OUT for the obvious non-W3C asks.
    "italian restaurant", "chinese restaurant", "japanese restaurant",
    "good restaurant", "best restaurant", "recommend a restaurant",
    "what's the weather", "what is the weather", "weather in",
    "what's the time", "what is the time",
    "translate this", "translate to",
    "write me a poem", "write a poem",
    "write me a song", "write a song",
    "summarize this article", "summarize this text",
    "react framework", "best framework", "best library",
    "best programming language", "vs code", "vscode",
    # Recipe / cooking already covered by ``cooking recipe`` /
    # ``cooking process recipe`` above.
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
