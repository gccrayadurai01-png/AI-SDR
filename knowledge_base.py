"""
Knight AI SDR — Dynamic knowledge base.
Product knowledge is loaded from agent scripts and uploaded documents.
"""

CLOUDFUZE_KNOWLEDGE = """
PRODUCT KNOWLEDGE:
This agent's knowledge comes from the active agent script configuration.
Upload documents via the Knowledge Base settings to add product-specific information.
The AI will use the value proposition, discovery questions, and objection handling
from the active agent's script to conduct natural conversations.
""".strip()


# --- Research report topics are now loaded dynamically from uploaded KB docs ---
RESEARCH_KB_TOPICS: list[tuple[str, str]] = [
    (
        "Sandler Selling Methodology",
        """The Knight AI SDR uses the Sandler selling methodology:
1. BONDING & RAPPORT: Build trust with a casual, low-pressure opener.
2. UP-FRONT CONTRACT: Set expectations for the call — "I'll ask a couple questions, if it doesn't fit, totally cool."
3. PAIN: Discover real pain points — ask about their world, what's broken, what costs them.
4. BUDGET: After establishing pain — "Is that something with budget allocated?"
5. DECISION: Understand the decision process — "Who else weighs in on this?"
6. FULFILLMENT: Connect their pain to your solution briefly.
7. POST-SELL: Lock the meeting, send invite, thank them.

STYLE RULES:
- 1-2 sentences max per turn. ONE question at a time.
- Exchange value: they share something, you give a relevant insight, then next question.
- Curious not scripted. Use natural acknowledgments.
- Never list features. Never monologue.
- Handle objections with empathy: Busy=when's better. Not interested=one probe then respect.""",
    ),
    (
        "Voice Realism & Pacing",
        """Voice call best practices for AI SDR:
- Keep responses to 1-2 sentences (~10-12 seconds of speech max)
- Pause 1-2 seconds between turns for natural rhythm
- Use conversational fillers sparingly: "Gotcha", "Makes sense", "Interesting"
- Reflect prospect's words back briefly before adding your insight
- One question at a time — never stack questions
- If they ask you something, answer briefly and honestly, then steer back
- Sound like a real person: vary tone, don't read lists
- Low-pressure CTAs: "Would it be crazy to take 15 minutes?" """,
    ),
    (
        "Objection Handling Framework",
        """OBJECTION MATRIX — acknowledge first, then one probe or pivot:

| Objection | Response Style |
|-----------|---------------|
| Not interested | "Totally fair" — one light probe about their current approach |
| Too busy | Respect it — offer specific short callback window |
| Send me an email | "Can do" — but prefer 15 min since every situation is different |
| Call me back later | Agree — lock a day/time, ask one qualifying question first |
| We already have a tool | Validate — ask what gaps they still see |
| No budget | Empathize — reframe as investment that pays back |
| You're a vendor | Light humor OK — ask permission for one question |
| Prove ROI | Offer demo or specialist — don't invent numbers |

Key: stay professional. If they shut down hard, leave the door open politely.""",
    ),
]


def _research_kb_text() -> str:
    return "\n\n".join(f"=== {title} ===\n{body}" for title, body in RESEARCH_KB_TOPICS)


# Optional: loaded from uploaded documents (API compatibility)
UPLOADED_DOCS_KNOWLEDGE: list[str] = []


def get_full_knowledge() -> str:
    """Base methodology + research topics + any uploaded doc snippets."""
    parts = [CLOUDFUZE_KNOWLEDGE, _research_kb_text()]
    parts.extend(UPLOADED_DOCS_KNOWLEDGE)
    return "\n\n".join(parts)
