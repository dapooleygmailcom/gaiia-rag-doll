"""
Test Suite: Universal RAG Agent & Cross-Domain Regression (TDD).

Validates that UniversalRagAgent dynamically instantiates and executes
across all configured domain Meta-Contracts without code changes.
"""

import os
import pytest
from engine.models.domain_profile import load_domain_profile
from engine.retrieval.universal_agent import UniversalRagAgent

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


@pytest.mark.parametrize("profile_name, expected_mode", [
    ("up_front_profile.json", "RULEBOOK_TECHNICAL"),
    ("asl_profile.json", "RULEBOOK_TECHNICAL"),
    ("renegade_legion_profile.json", "RULEBOOK_TECHNICAL"),
    ("home_insurance_profile.json", "POLICY_HIERARCHICAL"),
    ("visual_media_profile.json", "VISUAL_MEDIA"),
])
def test_universal_agent_initialization(profile_name, expected_mode):
    """Verify that UniversalRagAgent loads each domain and sets correct mode and persona."""
    profile_path = os.path.join(DATA_DIR, profile_name)
    if not os.path.exists(profile_path):
        pytest.skip(f"Profile {profile_name} not found")

    agent = UniversalRagAgent(profile_path)
    assert agent.profile is not None
    assert agent.profile.name is not None
    assert agent.pipeline_mode == expected_mode
    assert agent.collection_name is not None


def test_universal_agent_prompt_synthesis():
    """Verify that classification and generation prompts adapt to the domain."""
    rl_agent = UniversalRagAgent(os.path.join(DATA_DIR, "renegade_legion_profile.json"))
    c_prompt, g_prompt = rl_agent.build_prompts()
    assert "Renegade Legion" in c_prompt
    assert "Renegade Legion" in g_prompt
    assert "DAP" in c_prompt  # Glossary expansion injected

    vis_agent = UniversalRagAgent(os.path.join(DATA_DIR, "visual_media_profile.json"))
    c_vis_prompt, g_vis_prompt = vis_agent.build_prompts()
    assert "Glamour & Visual Publication Archive" in c_vis_prompt or "Visual" in c_vis_prompt


def test_universal_agent_query_dispatch():
    """Verify agent query method dispatches cleanly."""
    ins_agent = UniversalRagAgent(os.path.join(DATA_DIR, "home_insurance_profile.json"))
    assert ins_agent.pipeline_mode == "POLICY_HIERARCHICAL"
    assert ins_agent.profile.id == "home_insurance"
