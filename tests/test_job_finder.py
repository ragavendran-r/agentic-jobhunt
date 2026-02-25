"""
Tests for Job Finder Agent (CrewAI)
Run: pytest tests/test_job_finder.py -v
"""

import pytest
from unittest.mock import patch, MagicMock


# ── Unit Tests ────────────────────────────────────────────────────────────────


class TestSearchTool:
    """Tests for the Tavily search tool wrapper."""

    def test_extract_source_linkedin(self):
        from tools.search_tool import _extract_source

        assert _extract_source("https://linkedin.com/jobs/123") == "LinkedIn"

    def test_extract_source_naukri(self):
        from tools.search_tool import _extract_source

        assert _extract_source("https://naukri.com/job-listings") == "Naukri"

    def test_extract_source_unknown(self):
        from tools.search_tool import _extract_source

        assert _extract_source("https://someportal.com/job") == "Other"

    @patch("tools.search_tool.TavilyClient")
    def test_search_jobs_returns_normalized(self, mock_tavily):
        """Search results are normalized to expected format."""
        mock_client = MagicMock()
        mock_tavily.return_value = mock_client
        mock_client.search.return_value = {
            "results": [
                {
                    "title": "Engineering Manager at Freshworks",
                    "url": "https://linkedin.com/jobs/123",
                    "content": "We are hiring an EM with Golang experience...",
                    "score": 0.95,
                }
            ]
        }

        from tools.search_tool import search_jobs

        results = search_jobs("Engineering Manager", "Chennai")

        assert len(results) == 1
        assert results[0]["title"] == "Engineering Manager at Freshworks"
        assert results[0]["source"] == "LinkedIn"
        assert results[0]["score"] == 0.95

    @patch("tools.search_tool.TavilyClient")
    def test_search_jobs_empty_results(self, mock_tavily):
        """Empty search results handled gracefully."""
        mock_client = MagicMock()
        mock_tavily.return_value = mock_client
        mock_client.search.return_value = {"results": []}

        from tools.search_tool import search_jobs

        results = search_jobs("Engineering Manager", "Chennai")
        assert results == []


class TestJobFinderAgent:
    """Integration-style tests for the Job Finder CrewAI agent."""

    @patch("agents.job_finder.build_crew")
    def test_run_job_finder_returns_dict(self, mock_build_crew):
        """run_job_finder returns expected dict structure."""
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = '[{"title": "EM", "company": "Freshworks", "url": "http://test.com", "description": "test", "source": "LinkedIn", "fit_reason": "Good match"}]'
        mock_build_crew.return_value = mock_crew

        from agents.job_finder import run_job_finder

        result = run_job_finder(
            role="Engineering Manager",
            location="Chennai",
            tech_stack=["Golang", "AWS"],
            min_salary=7000000,
        )

        assert "jobs" in result
        assert "total_found" in result
        assert "top_matches" in result
        assert isinstance(result["jobs"], list)

    @patch("agents.job_finder.build_crew")
    def test_run_job_finder_handles_invalid_json(self, mock_build_crew):
        """run_job_finder handles non-JSON crew output gracefully."""
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "Here are the jobs: Freshworks, Chargebee"
        mock_build_crew.return_value = mock_crew

        from agents.job_finder import run_job_finder

        result = run_job_finder("Engineering Manager", "Chennai", ["Golang"], 5000000)

        assert "jobs" in result
        assert len(result["jobs"]) > 0
        assert "raw_output" in result["jobs"][0]
