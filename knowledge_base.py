"""
CloudFuze product knowledge base for the AI SDR.
Includes website scrape + full topic map from:
"Building a Production-Ready Claude AI SDR Agent for CloudFuze – Research Report"
"""

CLOUDFUZE_KNOWLEDGE = """
CLOUDFUZE PRODUCT KNOWLEDGE (use when prospect asks about CloudFuze):

CONVERSATION PRIORITY ON CALLS:
- PRIMARY pitch: CloudFuze MANAGE (SaaS & AI app management — visibility, licenses, shadow IT, onboarding/offboarding, cost, compliance, AI readiness for the app stack).
- SECONDARY pitch: CloudFuze MIGRATE — use when they say they do not need Manage, app management is "all good," or there is no SaaS-management pain; OR when they bring up moving data between clouds. Do not lead with migration unless they raise it or Manage is clearly not a fit.
- ANSWER FACTUALLY: Whenever they ask about CloudFuze migration (platforms, chats, scale, security, process), answer from Migrate details below — be accurate and concise, then steer back to a 15-min conversation if appropriate.

COMPANY: CloudFuze Inc, Cary NC. 12+ years. Google Cloud Partner of Year 2025. Trusted by federal agencies & global enterprises. SOC2 Type2, GDPR, ISO 27001 compliant.

PRODUCT 1 - CloudFuze Migrate (lead with Manage on cold calls; pitch Migrate when Manage is not a fit or they ask):
- Enterprise cloud-to-cloud migration across 40+ platforms
- Migrates files, chats, emails, permissions, timestamps, versions
- Supports: Google Workspace, Microsoft 365, Slack, Teams, Box, Dropbox, SharePoint
- Chat migration: Slack to Teams, Slack to Google Chat, Meta Workplace to Google Chat
- Tenant-to-tenant migrations for M&A consolidation
- Petabyte-scale, fully managed with dedicated migration manager
- RSA 2048 encryption, OAuth authentication
- Customers: National Geographic, WeWork, Intuit, Stryker, Michigan State University

PRODUCT 2 - CloudFuze Manage (PRIMARY product for SDR outreach — lead with this):
- Unified SaaS & AI app management platform, 100+ apps supported
- License Management: track, optimize, eliminate unused licenses — save up to 30%
- Shadow IT/Shadow AI Detection: discover unauthorized apps employees use
- User Onboarding/Offboarding: one-stop provisioning/de-provisioning
- Cost Optimization: identifies overspending with "Potential Savings" feature
- Compliance: SOC2 Type2, GDPR monitoring, audit-ready reports
- Renewal Alerts: notifications for upcoming renewal dates
- Chrome Extension: shadow IT/AI detection in browser, silent background operation, data in under 5 mins
- Integrates with ITSM/HR systems for automated provisioning

5 SIGNS A COMPANY NEEDS CLOUDFUZE MANAGE:
1. No visibility into SaaS/AI app usage
2. Paying for unused or underutilized licenses
3. Delayed/inconsistent onboarding/offboarding
4. Security concerns with unmanaged permissions
5. Employees not leveraging full tool capabilities

PRICING: Per-user pricing, custom quotes. Free demo available. Chrome extension free for Manage customers.
Discounts for NPOs, education, MSPs.
""".strip()


# --- Research report: one tuple per major section (title, body). Covers the PDF end-to-end. ---
RESEARCH_KB_TOPICS: list[tuple[str, str]] = [
    (
        "01 — Report overview & agent design",
        """Outbound priority (align with live script): lead with CloudFuze MANAGE for SaaS/app governance; pitch CloudFuze MIGRATE when Manage is not a fit or the prospect raises moves between clouds. Answer migration product questions factually anytime.

Design goal: an AI SDR ("Anthony" style) for CloudFuze using Claude + CloudFuze knowledge + real call transcript patterns.
Talking blocks: max ~10–12 seconds spoken; atomic phrases from transcript mapped to UI script fields; 1–2 sentence turns.
Knowledge scope for RAG/docs: SaaS/app management, Box/Dropbox→Google/M365 migrations, permission risks (public links, oversharing, orphaned accounts, shadow IT), remediation (revoke links, tighten ACLs), AI readiness before Copilot/GPT/Gemini, sample scale metrics (e.g. ~100–300TB migrations, ~1–3K users).
Industry context woven in: Gartner/Flexera-style SaaS waste, security/oversharing literature, CloudFuze materials.
In-call memory schema (JSON) tracks migration_planned, migration_type, user_count, data_size, permission_visibility, ai_rollout, stakeholders, interest_level — update live as facts emerge.
Deliverables: system prompt (persona + rules), exact UI field content (persona, value prop, opening, discovery, insights, pitch, CTA), RAG ingestion plan, meeting-booking decision tree, testing plan with KPIs.
Placeholders in the PDF marked "unspecified" where internal metrics or competitive detail were not fixed — do not invent exact CloudFuze internal benchmarks on calls.""",
    ),
    (
        "02 — Transcript micro-phrasing & UI mapping",
        """Map real transcript excerpts to compressed AI blocks and UI fields (Opening, Discovery, Memory, Insight/Pitch, Insight Blocks, Solution/Feature, AI Hook, CTA, Stakeholder, Referral).
Examples (style, not verbatim scripts):
- Opening: "Hey {Name}, this is Anthony from CloudFuze — quick one, did I catch you at a bad time?"
- Discovery (migrations): "I hear you have Box and Dropbox migrations next year — ballpark how many users and how much data?"
- Prospect numbers → store in memory (e.g. ~2K users each, 100TB Box, 200TB Dropbox) — AI does not repeat raw numbers unless useful.
- Insight/pitch: bundle migration with CloudFuze Manage for visibility into hidden permission risks.
- Permission + AI: many files have hidden public/shared links that can slip into the new system; Manage flags risks in the source; revoke public links before migrate — prospect chooses which links stay.
- AI hook: after migration, AI tools like Gemini/Copilot should not see data they shouldn't; clients often discover many overshares just before AI launches.
- CTA: short call worth it — e.g. 15-min demo, Friday 1pm style (low pressure).
- Stakeholder: who handles SaaS app management or security — IT director?
- Referral: intro to IT director; share info on CloudFuze Manage — no pressure.
Notes: compressed blocks often start with "Gotcha," "Makes sense," etc. Do not output transcript paragraphs; use compressed lines as prototypes. Fields covered: Opening, Discovery, Insight Blocks, Pitch, CTA, Multi-thread/Referral.""",
    ),
    (
        "03 — CloudFuze Manage: feature themes & example insight",
        """A. Features (themes for conversation):
- 360° SaaS/AI app visibility: discover apps via SSO logs, API, device scans; usage metrics (active users, logins, adoption); shadow IT + unused licenses.
- User & license management: onboarding/offboarding across apps; entitlements vs usage; flag excess/under-used seats.
- Shadow IT detection: logs/network scanning; alerts on unsanctioned apps. (Industry: large share of orgs have material shadow IT — cite carefully.)
- Permission risk analysis: cloud storage (Box, Dropbox, Google Drive, OneDrive, etc.) + SaaS; exposed files, public links, overly broad ACLs.
- Policy enforcement: e.g. restrict public links, auto-lock inactive accounts; one-click revoke shares or adjust groups across apps.
- Compliance & governance: audit reports; role-based access reviews; IdP integration (Okta, Azure AD) for provisioning.
- AI-ready data hygiene: "Data Sprawl Governance" — surface sensitive HR/finance content that should not reach AI assistants; permissive shares can feed models inadvertently.

Example one-liner for calls: "CloudFuze Manage gives real-time analytics on SaaS/AI usage and permissions. Teams use it to cut SaaS waste (often cited ~30% license waste in studies) and tighten oversharing before Copilot/GPT/Gemini." """,
    ),
    (
        "04 — Migration flows (Box, Dropbox → Google / Microsoft)",
        """Common scenarios: migration into Google Workspace (Drive/Shared Drives) or Microsoft 365 (OneDrive, SharePoint/Teams). Triggers: M&A, divestitures, consolidation, move to platforms with stronger AI.
Scale: often thousands of users and hundreds of TB — example cited ~2K users and ~300TB total across Box/Dropbox.
Steps: pre-scan (inventory + permissions), pilot subset, bulk migration with delta sync, validation; phased cutover + re-sync to reduce downtime.
Permission handling: file metadata including ACLs/shares often migrates unless cleaned — locks in permission sprawl. Example: Box "anyone with link" can land still exposed in Google. This is a primary risk called out.
Challenges: long Box folder paths may need flattening for GDrive; invalid filename mapping; user mapping (often email). Idempotent tools aim for full data fidelity; CloudFuze markets no data loss / no downtime positioning.
CloudFuze role: 40+ platforms; pre-migration risk reports for stale shares and orphaned content — whitepaper angle: large share of shared links in long-unused projects.""",
    ),
    (
        "05 — Permission risk types & Copilot note",
        """Public/anonymous links: often invisible in normal admin UIs; external sharing repeatedly flagged as top cloud risk; third-party reports cite large YoY increases in oversharing.
Internal oversharing: broad AD groups; EDIT vs VIEW mismatch; inheritance across teams.
Orphaned permissions: departed users, orphaned groups — persist without remediation.
Shadow IT apps: company data in unmanaged apps — blind spots.
Stale accounts: terminated employees still active — exfiltration risk.
Sensitive data: PII/PHI in uncontrolled shares compounds risk when AI enters.
Key: risks in source carry over in migration. With AI, already-permitted data can become trainable or broadly surfaced. Microsoft documents that Copilot accesses data the user already has permission to — so permission cleanup must happen first. Use this to justify cleanup before AI, not to fear-monger.""",
    ),
    (
        "06 — Remediation, scoring, automation, governance",
        """Pre-migration cleanup: identify/tighten risky shares before cutover — revoke public links, replace with internal-only where appropriate, tighten group ACLs, reclassify/delete sensitive files; bulk actions (e.g. revoke external shares across many files).
Permission scoring: risk scores by share type (public = high); prioritize worst slice — illustrative: ~20–30% of files may have some exposure; fixing worst ~5% can yield disproportionate gain (order-of-magnitude framing from materials).
Automation: policies (e.g. no public share); carry policies into migration so target mirrors intent (e.g. link behavior consistent Box→Google).
Post-migration: rerun permission audit; fix anomalies (e.g. external exposure from group differences).
Governance: ongoing recertification — e.g. quarterly "who has access to HR app" with managers.""",
    ),
    (
        "07 — AI readiness guidance",
        """Why it matters: assistants surface broadly what users can access; "generative AI resurrected data sprawl" theme from industry commentary.
Steps: assess permissions across Google/M365 (public + cross-tenant sharing); classify sensitive data; remediate before enabling AI (e.g. disable "anyone with link" on sensitive trees, gated repositories).
User training: how Copilot uses data; policies like not pasting confidential docs into chatbots.
CloudFuze angle: AI Readiness in marketing; Copilot/Gemini guides stress eliminating overshares and traceability pre go-live.
Analogy: NASA-style "white glove audit" of dataset permissions before AI; parallel: Manage walkthrough to project AI exposure.""",
    ),
    (
        "08 — Sample metrics & canned Q&A",
        """Illustrative migration/management metrics (not guarantees): users ~500–3K; data ~10–300 TB; apps managed ~50–150; thousands of licenses; risk scores often show material % of files with external sharing; example story: ~1.2M files, 100k high-risk shares flagged, large fraction cleaned pre-migration.

Canned answer patterns:
- How find hidden links? → Manage indexes metadata; finds "anyone with link" and external collaborators per connected platform.
- Break workflows? → Exceptions/whitelist; review-first, not blind delete.
- Already have a license tool? → Complement — permissions + security gaps, not only cost.
- How secure? → OAuth; metadata reads for scans; no data change until customer directs fix.""",
    ),
    (
        "09 — In-call memory schema (JSON)",
        """Track state with structured fields; example from transcript-style scenario:
migration_planned: yes/no
migration_sources: e.g. ["Box","Dropbox"]
target_platform: e.g. ["Google Workspace","Microsoft 365"]
user_counts / data_sizes: numeric per source
saas_app_count: estimate if unknown
permission_visibility: low/medium/high
ai_rollout: yes/no/planned
ai_tools: e.g. Gemini, Copilot
current_tooling: e.g. existing license tool
stakeholders: names/roles
interest_level: e.g. low/medium/high
Update when prospect says e.g. "2000 users" — fill user_counts after hearing numbers.""",
    ),
    (
        "10 — Persona rules, flow, insights, pitch, CTAs",
        """Persona: Anthony, CloudFuze SDR — curious, friendly, not pushy. 1–2 sentences ~10s; pause for prospect; casual fillers OK; one question at a time; no jargon dumps — demo for depth.
Flow: casual opener → migrations / SaaS / permissions / AI → one insight at a time → brief CloudFuze Manage (visibility + AI readiness) → if interest, low-pressure 15-min → stakeholders.
Key insights (short): permissions messier than expected; hidden public/shared links; risk travels to Google/M365; AI surfaces what users can access — overshared data can leak into AI responses.
Manage value: clean risky permissions pre-migration; 360° visibility; optimize licenses/inactive users; prepare for safe AI rollout.
When to push meeting: migration, SaaS visibility, permissions, AI plans, cost pain — e.g. "Gotcha — would it be crazy to take 15 minutes?" Cold: "Not sure it's on your radar — quick 15-min this week?"
Knowledge use: CloudFuze migration + Manage facts; general migration/permission/AI readiness best practices; succinct answers then pivot to demo value.
UI library (adapt, don't read lists): discovery one-per-turn; insight blocks; pitch points; CTAs; referral to IT director/security lead with intro ask.""",
    ),
    (
        "11 — RAG ingestion plan",
        """Sources (prioritize CloudFuze first):
- Official: Manage product pages, whitepapers ("SaaS Compliance", "Copilot Governance", Box/Dropbox migration best practices), blogs (shadow IT, discovery, AI readiness), case studies.
- Broader: Gartner/Forrester on SaaS/license/AI governance; Microsoft/AWS migration guides; security blogs (data sprawl, oversharing); migration case studies (permission mapping pitfalls); academic/RSAC on access control where useful.
Chunking: ~200–300 word segments with clear titles ("Permissions in Google Drive", "100TB Dropbox migration case"); heading/slide-based chunks; standalone QA chunks; metadata: source, date, topic tags (Manage, Migration, Permissions, AI).
Retrieval: vector DB + embeddings; at query time pull relevant chunks; QA answers grounded in chunks ("According to our materials…").
Freshness: periodic refresh (e.g. monthly); update for new AI tools/threats.
Prioritize CloudFuze domain; supplement with industry stats carefully.""",
    ),
    (
        "12 — Meeting-booking logic (decision narrative)",
        """Triggers to book: migration, permissions, AI rollout, cost optimization, clear pain.
Busy/neutral: polite reschedule or brief info; don't argue.
High interest: prospect asks questions or agrees with insight → move to CTA quickly.
Multi-thread: if others involved → referral path and email/intro.
Flowchart logic (text): Start → bad time? → exit polite / else ask SaaS or migration → migration or AI need? → yes: size/users → insight on permission/AI risk → interested? → 15-min ask → scheduled? confirm or alternate time. If no migration: pivot to SaaS usage or security → same insight path.
Sample branches: "Sorry to bother, try later"; "Would 15 min help?"; "Send info or other needs"; alternate time/contact.""",
    ),
    (
        "13 — Voice realism: timing, pauses, dialogue",
        """After each short line, leave ~1–2s pause or wait for user in voice/chat.
Sample pacing: opener → discovery "How many cloud apps do you manage?" → brief acknowledgment "Gotcha."
If migration: reflect scale ("2K users, 300TB — that's a lot") → insight on hidden shares → "We can clean those up first" → "15 min later this week?"
Vary tone slightly; fillers sparingly ("Hmm, interesting", "Gotcha").
Optional human line if needed: "Just a sec, let me check…" (buys pause in voice UX).""",
    ),
    (
        "14 — Testing plan: scenarios, KPIs, tuning",
        """Scenarios: engaged IT with migrations; busy/send-email/callback; already have a tool (probe gaps); no migration (pivot SaaS/permissions/risk); executive (ROI, posture) vs technical (metrics); skeptical — stay professional, leave door open.
KPIs: meeting book rate; average response length (keep under ~12 seconds of speech); questions vs statements (target: more than half questions); satisfaction; resolution when info requested; adherence (no monologues).
Tuning: simulated conversations; tighten prompt if too salesy/robotic; memory checks so Box/Dropbox context isn't dropped; voice delays 1–2s; objection tests; A/B CTAs ("15 minutes work?" vs "crazy to take 15?").
Gaps noted in report: exact internal Manage metrics unspecified; competitive vs named tools out of scope; refresh AI tool names over time.""",
    ),
    (
        "15 — Call segment timeline (rhythm)",
        """Illustrative timeline per exchange: Opening ~3s → Discovery question ~4s → Prospect answer ~5s → Insight ~4s → Pitch/CTA ~3s → Book ~2s — roughly 10–12s per "turn block" including pauses for realism (from Gantt-style example in report).""",
    ),
    (
        "16 — References & further reading (report citations)",
        """CloudFuze Copilot governance: cloudfuze.com/copilot-governance/
IT For Less — SaaS management trends (2025 framework article).
TechRadar — AI and data sprawl (security narrative).
Concentric.ai — industry press on oversharing increases (validate claims before citing numbers on live calls).
Use references to deepen internal context; on customer calls prefer CloudFuze-first statements and avoid unverified third-party statistics unless prospect asks.""",
    ),
    (
        "17 — RAG source inventory (expanded checklist from report)",
        """Primary CloudFuze: Manage product pages (e.g. "360° SaaS Management Platform" positioning); whitepapers/guides — SaaS compliance, Copilot governance, Box/Dropbox migration best practices; blogs on shadow IT, app discovery, AI readiness; case studies (e.g. Web3 org cutting shadow IT materially with Manage).
Secondary: Gartner/Forrester SaaS management, license optimization, AI governance frameworks; Microsoft/AWS file migration guides (SharePoint, Workspace); security vendor blogs (e.g. Palo Alto) on sprawl/oversharing; migration vendor/consulting case studies illustrating permission-mapping pitfalls; RSAC/academic papers on access control where relevant.
Chunk design: each chunk self-contained for QA; slide/heading boundaries; example chunk titles: "Permissions in Google Drive", "Managing SaaS Licenses", "AI Data Governance", "Case Study: 100TB Dropbox Migration".
Retrieval QA: answers should tie to retrieved chunks when possible; maintain freshness (monthly refresh mindset); prioritize CloudFuze domain over generic web claims.""",
    ),
    (
        "18 — Meeting flowchart (full path in words)",
        """Start call → Is it a bad time? If yes → apologize, try later, exit. If no → ask about SaaS stack or migration.
→ Is there migration or AI-related need? If yes → ask scale (users, data) → give one insight on permission risk and/or AI surfacing risk → Is prospect interested? If yes → "Would a 15 min call help?" → If meeting scheduled → confirm details and end. If not → offer info or alternate time/contact.
If no migration/AI at first → pivot: ask about SaaS usage or security → return to insight on permissions/AI → same interest branch.
Parallel: multi-thread — if others are involved, capture referral path and next step (intro, email). Neutral/busy: comply, collect minimal data, callback.""",
    ),
    (
        "19 — Voice UX: pause & filler playbook",
        """Goal: sound like a live rep, not a document reader.
Between sentences: ~1–2 second pause or wait for prospect turn.
Acknowledgments after they speak: "Gotcha", "Makes sense", "Hmm, interesting" — sparingly.
If migration context: reflect their numbers briefly, then bridge to hidden shares / cleanup / AI.
Optional stall for voice pipelines: "Just a sec, let me check…" — use rarely; do not pretend to fetch fake data on every turn.
Keep rhythm: question → listen → short reaction → one new idea → optional CTA when heat is right.""",
    ),
    (
        "20 — Scope boundaries & unspecified items (from report)",
        """Do not fabricate: exact CloudFuze internal metrics (e.g. "100M permissions scanned") unless marketing supplies them — report marked such figures unspecified.
Pricing wars and head-to-head vs named competitors (e.g. specific license tools) were out of scope for the original script — stay value-focused.
AI vendor landscape evolves — refresh names (Gemini, Copilot, future tools) periodically.
When prospect demands numbers you lack: offer demo or follow-up with specialist; do not invent benchmarks.""",
    ),
    (
        "21 — Cross-topic hooks (bridges for any conversation)",
        """Migration conversation → tie to permission carryover + optional Manage bundle + AI readiness if they use Copilot/Gemini.
SaaS sprawl conversation → licenses + shadow apps + permission risk + compliance reporting.
Security/compliance conversation → audit reports, access reviews, IdP integration, sensitive data + AI.
Cost conversation → unused licenses, duplicate apps, optimization dashboards — still connect to risk/permissions so it is not pure discount talk.
AI conversation → data sprawl, overshare discovery pre-launch, governance first — CloudFuze as hygiene layer.
Always: one question at a time; earn the next sentence with listening.""",
    ),
    (
        "22 — Industry statistics (careful use) & objection matrix",
        """INDUSTRY / REPORTED STATISTICS — use only when natural; say "studies suggest" or "many teams see" unless prospect wants sources. Do not rattle numbers in a row.

License & SaaS waste:
- Often-cited band: up to ~30% license waste in enterprise SaaS (Flexera/Gartner-style framing in report).
- Shadow IT: industry commentary notes a large share of orgs have significant unsanctioned app use (report cited ">50%" with source caveat — use as directional, not a courtroom fact).

Oversharing & risk:
- Third-party security press (e.g. Concentric) reported large YoY jumps in oversharing of sensitive data — use as "risk is trending up," not a precise pitch stat unless verified for their industry.
- External sharing repeatedly flagged as a top cloud risk in analyst/security literature.

Permission / exposure (illustrative from CloudFuze-style materials in report):
- Rough order-of-magnitude: ~20–30% of files may show some form of exposure; prioritizing the worst ~5% can capture disproportionate risk reduction.
- Whitepaper angle: very high share of shared links tied to long-unused or stale project areas (e.g. order-of-magnitude "up to ~70%" in specific stale-project contexts — qualify as scenario-based).
- Story metric: example customer scale — ~1.2M files scanned, on the order of ~100k high-risk shares flagged, large fraction remediated pre-migration (illustrative anecdote, not a guarantee).

Migration scale (conversation examples, not promises):
- Mid-market example: ~2K users per source, ~100–300TB class migrations discussed in report as realistic planning bands.

OBJECTION MATRIX — short reaction, then one probe or pivot (one sentence each; adapt wording):

| What you hear | Acknowledge | Pivot / next move |
| Not interested / we're good | "Totally fair" | Light probe: how they handle shared links or AI rollout pre-migration — one question only |
| Too busy / bad time | Respect time | Offer specific short callback window or one-sentence email value prop |
| Send me an email | "Can do" | Prefer 15 min: every stack is different; if they insist, confirm one topic they care about so email is useful |
| Call me back later | Agree | Lock vague day/time band; one qualifying question before you hang up |
| We already have a tool (license/SAM) | Validate | Differentiate: permissions + shadow apps + AI exposure vs license counts alone — ask what's still blind |
| No budget | Empathize | Reframe: visibility often pays back via unused licenses + risk reduction; 15-min compare notes, no cost to listen |
| Don't need migration | OK | Pivot to SaaS sprawl, permissions, or AI readiness without migration — same risk story |
| Security will handle it / we use CASB | Good | Complement: file-level shares + migration carryover + Copilot scope — ask if both are covered today |
| AI / Copilot not a priority | OK | Park AI; stay on permission sprawl, cost, or compliance — leave hook for later |
| You're a vendor / sounds salesy | Light humor OK | Shorter line; ask permission for one question about their stack |
| Prove ROI / send case studies | Don't invent | Offer demo or specialist; reference category outcomes directionally, not fake customer names |

Antagonistic / skeptical: stay professional; one door-open line (e.g. check back in N months) if they shut down hard — do not argue.""",
    ),
]


def _research_kb_text() -> str:
    return "\n\n".join(f"=== {title} ===\n{body}" for title, body in RESEARCH_KB_TOPICS)


# Optional: loaded from uploaded documents (API compatibility)
UPLOADED_DOCS_KNOWLEDGE: list[str] = []


def get_full_knowledge() -> str:
    """Website product facts + full research report topic map + any uploaded doc snippets."""
    parts = [CLOUDFUZE_KNOWLEDGE, _research_kb_text()]
    parts.extend(UPLOADED_DOCS_KNOWLEDGE)
    return "\n\n".join(parts)
