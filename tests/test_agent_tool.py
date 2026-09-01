import unittest

from app.agent_tool import KNOWLEDGE_BASE_SEARCH_TOOL, make_knowledge_base_search


class _FakeKnowledgeBase:
    def __init__(self):
        self.calls = []

    def ask(self, query, user_id):
        self.calls.append((query, user_id))
        return {
            "answer": "员工手册要求使用 MFA。[1]",
            "refusal": False,
            "citations": [1],
            "evidence": [{
                "rank": 1,
                "document_id": "handbook-id",
                "label": "员工手册 · 账号安全",
                "text": "所有员工必须启用 MFA。",
            }],
        }


class KnowledgeBaseSearchToolTests(unittest.TestCase):
    def test_schema_does_not_allow_model_to_supply_user_id(self):
        parameters = KNOWLEDGE_BASE_SEARCH_TOOL["function"]["parameters"]
        self.assertEqual(KNOWLEDGE_BASE_SEARCH_TOOL["function"]["name"], "knowledge_base_search")
        self.assertNotIn("user_id", parameters["properties"])
        self.assertFalse(parameters["additionalProperties"])

    def test_handler_binds_authenticated_user_and_preserves_sources(self):
        kb = _FakeKnowledgeBase()
        resolved = []

        def resolve(kb_id):
            resolved.append(kb_id)
            return kb

        tool = make_knowledge_base_search(resolve, "student")
        result = tool("MFA 是否必须启用？", "school")

        self.assertEqual(resolved, ["school"])
        self.assertEqual(kb.calls, [("MFA 是否必须启用？", "student")])
        self.assertEqual(result["kb_id"], "school")
        self.assertEqual(result["sources"][0]["citation"], 1)
        self.assertEqual(result["sources"][0]["document_id"], "handbook-id")

    def test_empty_user_is_rejected_before_tool_can_run(self):
        with self.assertRaisesRegex(ValueError, "缺少"):
            make_knowledge_base_search(lambda _kb_id: _FakeKnowledgeBase(), "  ")


if __name__ == "__main__":
    unittest.main(verbosity=2)
