# Hai Family Agent — `agent/` implementation workspace

This folder is where the Hai Family Agent is built. It is the Muse Code workspace.

The full product spec lives in the root [`README.md`](../README.md): one shared family companion, individual conversations per member, and memories that connect people without compromising privacy. This file mirrors its instructions for the implementation here. If the two disagree, the root README wins.

This is a standalone hackathon prototype. It does not connect to Hai's production backend, Oracle server, Firebase project, or user data. All demo data is fictional.

## What to build

- **Shared family agent:** one agent per family workspace. Members join via email or phone identifiers and each get a one-to-one conversation with the same agent. Prototype may simulate invitations and identities.
- **Individual conversations:** each member shares updates, asks questions, records memories, plans events, and requests reminders. Every request is handled under that member's identity and permissions.
- **Three memory categories**, enforced in application code and storage — never by model instructions alone:
  - **Private persona memory** — stays with the individual unless explicitly authorized for sharing.
  - **Family-shared memory** — explicitly approved for family-wide access (birthdays, anniversaries, relationships, events, shared preferences).
  - **Restricted information** — never disclosed without the required permission (surprise plans, private conversations, gifts, sensitive details).
- **Events and reminders:** identify upcoming birthdays, anniversaries, celebrations, and commitments; determine the right recipients; draft reminders without leaking restricted details to the wrong person.
- **Personalized messages:** draft recipient-appropriate messages using only information the requester is authorized to share; always present for review and require approval before any send or consequential action.

## Agent tools

One family agent calling tools — these are capabilities of the same agent, not separate agents:

| Tool | Responsibility |
|---|---|
| `share_agent_family` | Associate the shared agent with a family workspace and its members using email or phone identifiers, subject to the implemented identity and invitation mechanism. |
| `get_family_events` | Retrieve upcoming birthdays, anniversaries, and family events from authorized profiles and approved memory sources. |
| `get_due_reminders` | List proactive reminders for this member: due family events and public holidays, each with its recipients. |
| `retrieve_family_memories` | Search family memories relevant to the current request while enforcing family boundaries and memory visibility permissions. |
| `draft_family_message` | Draft a reminder or message for specified recipients using information they are authorized to receive. |
| `request_action_approval` | Present a proposed action, recipients, and message details for human review before execution. |

The backend validates tool inputs, enforces authorization, and verifies action results. The agent must not bypass access controls by requesting restricted information through another tool or model call.

## Family memory workflow

1. **Capture:** a member shares a fact through text or voice.
2. **Extract:** a Meta model identifies candidate people, relationships, dates, and memory categories.
3. **Confirm:** the agent asks the user to review the proposed memory and its visibility.
4. **Store:** the application validates and saves the approved memory in the dedicated family-memory store.
5. **Retrieve:** the agent searches authorized memories and upcoming events when a member asks.
6. **Coordinate:** the agent identifies relevant recipients and drafts reminders or messages.
7. **Approve:** an authorized human reviews the proposed action.
8. **Execute and verify:** if an approved delivery mechanism exists, execute and verify; otherwise demonstrate the proposed action without claiming it was sent.

## Proactive reminders

The agent checks `get_due_reminders` when a conversation starts: family events
in the window plus upcoming US public holidays. Birthdays skip the birthday
person, secrets stay among those in the know, and shared celebrations and
public holidays go to everyone. Each reminder is proposed once through a
`schedule_reminder` approval, and holidays move from a planning proposal to a
week-before reminder. Approving a reminder delivers a personalized invitation
card (with Muse Image art when available) to each recipient's People thread.

## Meta models used

| Model | What it does in Hai Family | File |
|---|---|---|
| Muse Wonder (`MODEL_AGENT`, default `muse-wonder-1.0`) | Interactive family agent: reasoning and tool coordination in each member conversation. Interactive calls use a 25s timeout with no retry; on timeout the loop falls back to `MODEL_FAST` immediately and records it in the run log | `family_agent/loop.py` |
| Muse Spark (`MODEL_FAST`, default `muse-spark-1.3`) | Fast fallback for family chat, and the setting for short text jobs (reminder copy, suggested replies, translation) | `family_agent/loop.py`, `agent_kit/settings.py` |
| Muse Spark (`MODEL_VERIFY`, default `muse-spark-1.3`) | Verification judges for claims and checklist checks | `verification/verify_claims.py`, `verification/verify_checklist.py` |
| Muse Spark (`MODEL_API_MODEL`, default `muse-spark-1.3`) | Fallback default for callers without a task-specific setting (legal agent, document jobs, run records) | `agent_kit/settings.py`, `legal_agent/loop.py` |
| Muse Image (`IMAGE_MODEL`, default `muse-image-1.0`) | Invitation card art for approved reminders (CSS card fallback) | `family_agent/invitations.py` |
| SAM 3.1 (`SAM_MODEL`, default `sam-3.1`) | Legal-side signature segmentation only; not used by the family agent | `agent_kit/tools/check_signatures.py` |
| Muse Voice Transcribe (`TRANSCRIBE_MODEL`, default `muse-voice-transcribe-1.0`) | Configured for voice memory capture; no voice input is wired up yet | `agent_kit/settings.py` |

The OpenAI Python SDK is used only as an HTTP client pointed at Meta's Model API (api.meta.ai/v1); no OpenAI models are used.

Live leak test (2026-11-02, Muse Wonder with `VERIFICATION=on`, Muse Spark judge, `./helper fleak`): 34/40 passed — Maya 20/23, Sofia 7/10, Eli 2/2, Leo controls 5/5; the 6 failures were judged as hinting at a secret, mostly "neither confirm nor deny" phrasing rather than direct disclosure. Full report: `workspace/evals/leak_test_2026-11-02.json`.

Still evaluations, not integrations: Muse Glimmer and Muse Spark Open (open-weight, separate inference setup). **Muse Code** is the development agent, not a user-facing family agent. A model counts as implemented only after its integration is tested.

## Trust, privacy, and verification

- Keep private, family-shared, and restricted information distinct; require confirmation before adding to shared memory.
- Scope every query to the correct family and requesting member; enforce authorization in code and storage operations.
- Require human approval before sending messages or consequential actions; verify delivery before claiming completion.
- Verify answers against retrieved information; use fictional demo data.
- Tests must cover authorized birthday reminders, private memories, restricted surprise plans, and cross-family access attempts.

## Prototype scope

In scope: one family workspace with multiple fictional members, separate 1:1 conversation contexts, three memory categories with permission checks, confirmed memory creation and persistent retrieval, event identification with recipient-specific drafting, human approval workflow, Meta model integration, and verification against improper disclosure.

Out of scope unless explicitly implemented: production authentication, real email/SMS delivery, scheduled push notifications, full multi-family deployment, and integration with the private Hai application.

## This folder (implementation map)

- `family_agent/` — the family agent: the conversation loop (`loop.py`), CLI (`__main__.py`), family-memory store (`store.py`), demo data (`demo_data.py`, `seed.py`), `system_instructions.md`, per-model status (`EVALS.md`), and `tools/` (one file per tool plus the registry).
- `agent_kit/` — reused plumbing: Model API client, run progress on disk, model settings.
- `tests/test_family_*.py` — offline family tests (scripted API, never live); `tests/family_support.py` — seeded demo family helpers.
- `workspace/families/` — family records; `workspace/runs/` — conversations.
- `legal_agent/`, `verification/`, `scripts/`, `world/` — legal-workshop code, kept intact.
- `.env` — needs `MODEL_API_KEY` for the Meta Model API (copy from `.env.example`; never commit the populated file).

## Setup and commands

```bash
cp .env.example .env      # add MODEL_API_KEY
./helper fseed   # seeds the riveras demo family (8 members, including grandparents, uncle and aunt)
./helper frun --family riveras --member maya --message "When is the next birthday?"
./helper fresume RUN_ID --message "And after that?"
./helper freview --family riveras --list
./helper fconsole                 # local demo console: http://127.0.0.1:8766
./helper test [--verification]   # offline tests; never call the live API
```
