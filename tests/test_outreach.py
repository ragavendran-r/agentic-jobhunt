"""
Tests for Outreach Agent (OpenAI Agents SDK)
Run: pytest tests/test_outreach.py -v
"""

import pytest
from unittest.mock import patch, MagicMock


# ── Unit Tests ────────────────────────────────────────────────────────────────


class TestOutreachTools:
    """Tests for the outreach agent tool functions."""

    def test_get_candidate_profile_returns_json(self):
        """get_candidate_profile returns valid JSON string."""
        import json
        from app_agents.outreach import get_candidate_profile

        result = get_candidate_profile()
        profile = json.loads(result)

        assert "name" in profile
        assert "title" in profile
        assert "key_skills" in profile
        assert "certifications" in profile
        assert profile["name"] == "Ragavendran Ramalingam"

    def test_draft_linkedin_message_contains_company(self):
        """LinkedIn message draft contains the target company name."""
        from app_agents.outreach import draft_linkedin_message

        result = draft_linkedin_message(
            company="Freshworks",
            role="Engineering Manager",
            hiring_manager="John",
            fit_reason="Strong Golang SaaS match",
        )

        assert "Freshworks" in result
        assert "Engineering Manager" in result
        assert "John" in result

    def test_draft_linkedin_message_no_manager(self):
        """LinkedIn message handles missing hiring manager gracefully."""
        from app_agents.outreach import draft_linkedin_message

        result = draft_linkedin_message(
            company="Chargebee",
            role="Engineering Manager",
            hiring_manager="",
            fit_reason="Good SaaS fit",
        )

        assert "Chargebee" in result
        assert len(result) > 50

    def test_draft_cover_letter_contains_key_info(self):
        """Cover letter contains company, role, and key skills."""
        from app_agents.outreach import draft_cover_letter

        result = draft_cover_letter(
            company="Kissflow",
            role="Engineering Manager",
            job_description="We need an EM with Golang and AWS experience",
            matching_skills=["Golang", "AWS", "Kubernetes"],
        )

        assert "Kissflow" in result
        assert "Engineering Manager" in result
        assert "Ragavendran" in result

    def test_draft_cover_letter_empty_skills(self):
        """Cover letter handles empty skills list gracefully."""
        from app_agents.outreach import draft_cover_letter

        result = draft_cover_letter(
            company="TestCo",
            role="EM",
            job_description="Generic EM role",
            matching_skills=[],
        )

        assert len(result) > 100


class TestOutreachAgent:
    """Tests for the outreach agent runner."""

    def test_run_outreach_empty_jobs(self):
        """run_outreach handles empty job list gracefully."""
        from app_agents.outreach import run_outreach

        result = run_outreach([], "Ragavendran")
        assert result["total_drafted"] == 0
        assert result["outreach"] == []

    @patch("app_agents.outreach.Runner")
    def test_run_outreach_returns_structure(self, mock_runner):
        """run_outreach returns expected dict structure."""
        mock_result = MagicMock()
        mock_result.final_output = "Draft LinkedIn message and cover letter for Freshworks..."
        mock_runner.run_sync.return_value = mock_result

        from app_agents.outreach import run_outreach

        sample_jobs = [
            {
                "company": "Freshworks",
                "title": "Engineering Manager",
                "url": "https://freshworks.com",
                "match_score": 85,
                "matching_skills": ["Golang", "AWS"],
                "strengths": "Great SaaS fit",
                "description": "EM role for SaaS team",
            }
        ]

        result = run_outreach(sample_jobs, "Ragavendran")

        assert "outreach" in result
        assert "total_drafted" in result
        assert result["total_drafted"] == 1
        assert result["outreach"][0]["company"] == "Freshworks"

    @patch("app_agents.outreach.Runner")
    def test_run_outreach_caps_at_five(self, mock_runner):
        """run_outreach processes max 5 jobs even if more provided."""
        mock_result = MagicMock()
        mock_result.final_output = "Drafted messages..."
        mock_runner.run_sync.return_value = mock_result

        from app_agents.outreach import run_outreach

        many_jobs = [
            {
                "company": f"Company{i}",
                "title": "EM",
                "url": f"http://co{i}.com",
                "match_score": 80,
                "matching_skills": [],
                "strengths": "",
                "description": "",
            }
            for i in range(8)
        ]

        result = run_outreach(many_jobs, "Ragavendran")
        assert result["total_drafted"] <= 5
