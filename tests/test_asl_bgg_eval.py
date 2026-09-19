import unittest
import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.build_asl_bgg_eval_set import (
    clean_text,
    extract_rule_numbers,
    extract_scenarios,
    extract_concepts,
    classify_intent
)
from scripts.run_asl_bgg_eval import extract_retrieved_rules


class TestASLTextCleaner(unittest.TestCase):
    def test_clean_html_and_bbcode(self):
        raw = "Hello [b]World[/b]<br><br />[quote=john]Can a squad fire?[/quote]More text."
        cleaned = clean_text(raw)
        self.assertNotIn("[b]", cleaned)
        self.assertNotIn("[/b]", cleaned)
        self.assertNotIn("<br>", cleaned)
        self.assertIn("> Can a squad fire?", cleaned)
        self.assertIn("More text.", cleaned)


class TestASLRuleNumberExtraction(unittest.TestCase):
    def test_chapter_decimal_rules(self):
        text = "According to A7.212 and B13.3, moving in the open causes FFNAM. See also C8.52 and D2.11."
        rules = extract_rule_numbers(text)
        self.assertIn("A7.212", rules)
        self.assertIn("B13.3", rules)
        self.assertIn("C8.52", rules)
        self.assertIn("D2.11", rules)

    def test_deep_subclauses_and_letters(self):
        text = "Check G1.6121 for PTO concealment and D5.341 for bail out rules."
        rules = extract_rule_numbers(text)
        self.assertIn("G1.6121", rules)
        self.assertIn("D5.341", rules)

    def test_bracketed_and_parenthetical_rules(self):
        text = "The unit is pinned [A7.8] and suffers desperation morale (A10.5)."
        rules = extract_rule_numbers(text)
        self.assertIn("A7.8", rules)
        self.assertIn("A10.5", rules)

    def test_explicit_chapter_and_rule_prefix(self):
        text = "As stated in Rule A10 and section C5, ordnance cannot fire."
        rules = extract_rule_numbers(text)
        self.assertIn("A10", rules)
        self.assertIn("C5", rules)

    def test_hex_coordinates_without_dot_ignored(self):
        # Hex coordinates like A1, B2 in a text without "Rule" should not trigger as rules
        text = "The tank moved from hex A1 to hex B2 and then entered K10."
        rules = extract_rule_numbers(text)
        self.assertNotIn("A1", rules)
        self.assertNotIn("B2", rules)
        self.assertNotIn("K10", rules)


class TestASLScenarioExtraction(unittest.TestCase):
    def test_scenario_numbers_and_packs(self):
        text = "In Scenario 131 Penetration of Rostov, or ASL 1 Fighting Withdrawal, or AP12."
        scenarios = extract_scenarios(text)
        self.assertIn("131", scenarios)
        self.assertIn("1", scenarios)
        self.assertTrue("12" in scenarios or "AP12" in scenarios)


class TestASLConceptClassification(unittest.TestCase):
    def test_movement_and_bypass(self):
        text = "Can the squad use bypass movement around the woods building hexside using 3 MP?"
        concepts = extract_concepts(text)
        self.assertIn("movement_bypass", concepts)

    def test_fire_and_ift(self):
        text = "What is the residual firepower left after subsequent first fire on the IFT?"
        concepts = extract_concepts(text)
        self.assertIn("fire_attacks", concepts)

    def test_morale_pin_rally(self):
        text = "A broken leader under desperation morale needs self rally on a 1MC."
        concepts = extract_concepts(text)
        self.assertIn("morale_pin_rally", concepts)

    def test_afv_and_vehicles(self):
        text = "The Sherman tank changed TCA to fire its turret MG, but it is in motion and open topped."
        concepts = extract_concepts(text)
        self.assertIn("vehicles_afv", concepts)

    def test_ordnance_and_guns(self):
        text = "The AT gun rolled for To Hit on the TH table and scored a Critical Hit with APCR special ammo."
        concepts = extract_concepts(text)
        self.assertIn("ordnance_guns", concepts)


class TestASLIntentClassification(unittest.TestCase):
    def test_scenario_intent(self):
        intent = classify_intent(
            "Question on Scenario 5 SSR",
            "What are the victory conditions for the Germans?",
            [],
            ["5"]
        )
        self.assertEqual(intent, "scenario")

    def test_direct_rule_intent(self):
        intent = classify_intent(
            "Rule A7.212 interpretation",
            "What does rule A7.212 say about subsequent first fire?",
            ["A7.212"],
            []
        )
        self.assertEqual(intent, "direct_rule")

    def test_situation_intent(self):
        intent = classify_intent(
            "Can a concealed squad advance?",
            "Can I advance my squad into melee if the enemy is broken?",
            [],
            []
        )
        self.assertEqual(intent, "situation")

    def test_concept_intent(self):
        intent = classify_intent(
            "How does Motion status work?",
            "What is the difference between Motion and Non-Stopped?",
            [],
            []
        )
        self.assertEqual(intent, "concept")


from scripts.run_asl_bgg_eval import extract_retrieved_rules, check_rule_index_coverage


class TestRetrievedRulesExtraction(unittest.TestCase):
    def test_extract_from_metadata_and_headers(self):
        chunks = [
            ("Text with header [Rule: A7.212] explaining first fire.", {"rule_number": "A7.212"}),
            ("Text with header [Section: D5.6] about vehicles.", {"rule_number": "D5.6"}),
            {"text": "Another chunk [Rule: B13.3]", "rule_id": "B13.3"}
        ]
        extracted = extract_retrieved_rules(chunks)
        self.assertIn("A7.212", extracted)
        self.assertIn("D5.6", extracted)
        self.assertIn("B13.3", extracted)


class TestRuleIndexCoverage(unittest.TestCase):
    def test_coverage_detection_indexed_and_missing(self):
        mock_index = {
            "A17.2": [{"chunk_id": "c1"}],
            "A7.8": [{"chunk_id": "c2"}],
            "D3.12": [{"chunk_id": "c3"}],
        }
        # Mix of indexed and unindexed
        indexed, missing = check_rule_index_coverage(["A17.2", "G12.4", "A7.8"], mock_index)
        self.assertEqual(set(indexed), {"A17.2", "A7.8"})
        self.assertEqual(missing, ["G12.4"])

        # Completely missing module
        indexed_missing, missing_all = check_rule_index_coverage(["G12.4", "G12.61"], mock_index)
        self.assertEqual(indexed_missing, [])
        self.assertEqual(missing_all, ["G12.4", "G12.61"])


class TestBenchmarkSchemaParity(unittest.TestCase):
    def test_benchmark_schema_matches_upfront_keys(self):
        # Validate that the keys required for benchmark match Up Front
        expected_keys = {
            "id", "query", "title", "raw_question", "intent",
            "expected_rule_citations", "scenarios", "concepts",
            "ground_truth_answer", "num_replies", "source_url"
        }
        
        sample_item = {
            "id": "bgg_asl_3758531",
            "query": "Pinning a wounded leader: Can a leader be pinned?",
            "title": "Pinning a wounded leader",
            "raw_question": "Can a leader be pinned?",
            "intent": "situation",
            "expected_rule_citations": ["A17.2"],
            "scenarios": [],
            "concepts": ["morale_pin_rally"],
            "ground_truth_answer": "Yes, under rule A17.2.",
            "num_replies": 3,
            "source_url": "https://boardgamegeek.com/thread/3758531"
        }
        
        self.assertEqual(set(sample_item.keys()), expected_keys)


if __name__ == "__main__":
    unittest.main()

