"""The Legal Agent (legal_agent/loop.py): the agentic loop, verification off."""

from __future__ import annotations

import unittest

from agent_kit.settings import verification_enabled
from legal_agent.loop import resume_run, run_agent
from tests import support
from tests.fake_model import Refused

CHECKLIST, PLAN, LIST, READ, FACT, MEMO = (
    support.CHECKLIST,
    support.PLAN,
    support.LIST,
    support.READ,
    support.FACT,
    support.MEMO,
)


@unittest.skipIf(
    verification_enabled(), "covered by tests/test_verification.py when VERIFICATION=on"
)
class AgentLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root, tmp = support.make_root()
        self.addCleanup(tmp.cleanup)

    def test_each_tool_runs_and_the_memo_ends_the_run(self) -> None:
        fake = support.scripted(self.root, [CHECKLIST, PLAN, LIST, READ, FACT, MEMO])
        result = run_agent(root=self.root, client=fake.client, turns=10)
        self.assertEqual("answered", result["status"])
        calls = support.tool_calls(self.root, result["run_id"])
        self.assertEqual(
            [
                "set_checklist",
                "update_plan",
                "list_documents",
                "analyze_document",
                "record_fact",
                "submit_memo",
            ],
            [name for name, _args, _result in calls],
        )
        results = {name: out for name, _args, out in calls}
        self.assertEqual(PLAN[1]["plan"], results["update_plan"]["plan"])
        self.assertIn("DOC-REPORT", str(results["list_documents"]))
        self.assertTrue(
            all(
                e["verified"]
                for d in results["analyze_document"]["documents"]
                for e in d["evidence"]
            )
        )
        self.assertEqual("ready_statement", results["record_fact"]["recorded"]["item"])
        case = support.case(self.root, result["run_id"])
        self.assertEqual("unverified", case["memo"]["status"])
        self.assertEqual("proven", case["checklist"]["ready_statement"]["status"])
        self.assertEqual(["DOC-REPORT", "DOC-MINUTES"], case["read_ids"])

    def test_a_refused_tool_comes_back_as_an_error_the_model_can_read(self) -> None:
        bad = ("record_fact", dict(FACT[1], quote="The vendor admitted it was broken"))
        fake = support.scripted(self.root, [CHECKLIST, READ, bad, MEMO])
        result = run_agent(root=self.root, client=fake.client, turns=10)
        refused = support.tool_calls(self.root, result["run_id"])[2][2]
        self.assertIn("quote not found", refused["error"])
        self.assertEqual("answered", result["status"])

    def test_checklist_is_not_checked_without_verification(self) -> None:
        fake = support.scripted(self.root, [support.PARTIAL_CHECKLIST, MEMO])
        result = run_agent(root=self.root, client=fake.client, turns=5)
        case = support.case(self.root, result["run_id"])
        self.assertEqual(4, len(case["checklist"]))
        self.assertEqual([], case["checklist_checks"])
        offered = {t["name"] for t in support.first_agent_request(fake)["tools"]}
        self.assertNotIn("request_evidence", offered)

    def test_instructions_come_from_the_behaviour_file(self) -> None:
        fake = support.scripted(self.root, [CHECKLIST, MEMO])
        run_agent(root=self.root, client=fake.client, turns=5)
        sent = support.first_agent_request(fake).get("instructions", "")
        self.assertEqual(support.instructions_file().strip(), sent.strip())

    def test_progress_is_saved_every_turn(self) -> None:
        fake = support.scripted(self.root, [PLAN, LIST, MEMO])
        result = run_agent(root=self.root, client=fake.client, turns=10)
        progress = support.progress(self.root, result["run_id"])
        self.assertEqual(3, progress["turn"])
        self.assertEqual([1, 2, 3], [t["turn"] for t in progress["turns"]])
        self.assertEqual("answered", progress["status"])

    def test_tool_result_is_fed_back_as_the_next_observation(self) -> None:
        fake = support.scripted(self.root, [LIST, MEMO])
        run_agent(root=self.root, client=fake.client, turns=10)
        second = support.agent_requests(fake)[1]
        outputs = [
            i for i in second["input"] if i.get("type") == "function_call_output"
        ]
        self.assertTrue(outputs and "DOC-REPORT" in outputs[-1]["output"])

    def test_plain_text_gets_a_nudge_to_call_a_tool(self) -> None:
        fake = support.scripted(self.root, [LIST, MEMO])
        original, replies = fake.responder, []

        def text_first(kind, payload):
            if kind == "respond" and "tools" in payload and not replies:
                replies.append(1)
                return {"output_text": "I think the vendor knew."}
            return original(kind, payload)

        fake.responder = text_first
        result = run_agent(root=self.root, client=fake.client, turns=10)
        self.assertEqual("answered", result["status"])
        self.assertIn(
            "Call exactly one tool", str(support.agent_requests(fake)[1]["input"][-1])
        )

    def test_turn_limit_pauses_and_resume_continues_the_same_run(self) -> None:
        fake = support.scripted(self.root, [PLAN, LIST, READ, MEMO])
        result = run_agent(root=self.root, client=fake.client, turns=2)
        self.assertEqual("paused", result["status"])
        resumed = resume_run(
            root=self.root, run_id=result["run_id"], client=fake.client, turns=5
        )
        self.assertEqual(
            (result["run_id"], "answered"), (resumed["run_id"], resumed["status"])
        )
        self.assertIn("more turns", str(support.agent_requests(fake)[2]["input"]))

    def test_an_answered_run_has_nothing_to_resume(self) -> None:
        fake = support.scripted(self.root, [MEMO])
        result = run_agent(root=self.root, client=fake.client, turns=5)
        before = len(fake.calls)
        again = resume_run(
            root=self.root, run_id=result["run_id"], client=fake.client, turns=5
        )
        self.assertEqual("answered", again["status"])
        self.assertEqual(before, len(fake.calls))

    def test_last_turn_asks_for_the_memo(self) -> None:
        fake = support.scripted(self.root, [PLAN, LIST, MEMO])
        run_agent(root=self.root, client=fake.client, turns=3)
        self.assertIn("submit_memo", str(support.agent_requests(fake)[-1]["input"][-1]))

    def test_a_model_api_failure_pauses_the_run_so_it_can_resume(self) -> None:
        fake = support.scripted(self.root, [LIST, MEMO])
        original, failures = fake.responder, []

        def outage(kind, payload):
            if (
                kind == "respond"
                and "tools" in payload
                and len(support.agent_requests(fake)) == 2
                and not failures
            ):
                failures.append(1)
                raise Refused("service_overloaded")
            return original(kind, payload)

        fake.responder = outage
        result = run_agent(root=self.root, client=fake.client, turns=10)
        self.assertEqual("paused", result["status"])
        self.assertEqual(
            "paused", support.progress(self.root, result["run_id"])["status"]
        )
        resumed = resume_run(
            root=self.root, run_id=result["run_id"], client=fake.client, turns=5
        )
        self.assertEqual("answered", resumed["status"])


if __name__ == "__main__":
    unittest.main()
