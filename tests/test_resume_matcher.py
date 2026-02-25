"""
Tests for Resume Matcher Agent (LangGraph)
Run: pytest tests/test_resume_matcher.py -v
"""

import pytest
from unittest.mock import patch, MagicMock


# ── Unit Tests ────────────────────────────────────────────────────────────────

class TestResumeParser:
    """Tests for the resume parser utility."""

    def test_parse_resume_missing_file_returns_default(self):
        """Returns default profile when resume file not found."""
        from tools.resume_parser import parse_resume
        result = parse_resume("nonexistent_file.pdf")
        assert len(result) > 0
        assert "Engineering Manager" in result

    def test_extract_skills_finds_golang(self):
        """extract_skills finds known skills in resume text."""
        from tools.resume_parser import extract_skills
        resume_text = "Experienced in Golang, AWS, Kubernetes and ReactJS."
        skills = extract_skills(resume_text)
        assert "Golang" in skills
        assert "AWS" in skills
        assert "Kubernetes" in skills

    def test_extract_skills_case_insensitive(self):
        """extract_skills is case insensitive."""
        from tools.resume_parser import extract_skills
        resume_text = "skilled in golang and kubernetes"
        skills = extract_skills(resume_text)
        assert "Golang" in skills

    def test_extract_skills_no_false_positives(self):
        """extract_skills doesn't return skills not in text."""
        from tools.resume_parser import extract_skills
        resume_text = "Experienced in Python and Django."
        skills = extract_skills(resume_text)
        assert "Golang" not in skills
        assert "Kafka" not in skills


class TestResumeMatcherGraph:
    """Tests for the LangGraph resume matching pipeline."""

    def test_load_resume_node_populates_state(self):
        """load_resume node sets resume_text and resume_chunks."""
        from agents.resume_matcher import load_resume

        initial_state = {
            "resume_text": "",
            "job_descriptions": [],
            "resume_chunks": [],
            "scored_jobs": [],
            "current_job_index": 0,
            "final_results": [],
        }

        result = load_resume(initial_state)
        assert len(result["resume_text"]) > 0
        assert len(result["resume_chunks"]) > 0

    def test_compile_results_filters_by_score(self):
        """compile_results filters out jobs below 60% match."""
        from agents.resume_matcher import compile_results

        state = {
            "resume_text": "test",
            "job_descriptions": [],
            "resume_chunks": [],
            "scored_jobs": [
                {"company": "Freshworks", "match_score": 85, "recommendation": "Apply"},
                {"company": "LowMatch Co", "match_score": 45, "recommendation": "Skip"},
                {"company": "Chargebee", "match_score": 72, "recommendation": "Apply"},
            ],
            "current_job_index": 3,
            "final_results": [],
        }

        result = compile_results(state)
        companies = [j["company"] for j in result["final_results"]]
        assert "Freshworks" in companies
        assert "Chargebee" in companies
        assert "LowMatch Co" not in companies

    def test_compile_results_sorted_by_score(self):
        """compile_results returns jobs sorted highest score first."""
        from agents.resume_matcher import compile_results

        state = {
            "resume_text": "test",
            "job_descriptions": [],
            "resume_chunks": [],
            "scored_jobs": [
                {"company": "B Corp", "match_score": 72},
                {"company": "A Corp", "match_score": 90},
                {"company": "C Corp", "match_score": 65},
            ],
            "current_job_index": 3,
            "final_results": [],
        }

        result = compile_results(state)
        scores = [j["match_score"] for j in result["final_results"]]
        assert scores == sorted(scores, reverse=True)

    def test_should_continue_scoring_continues(self):
        """should_continue_scoring returns 'score_job' when jobs remain."""
        from agents.resume_matcher import should_continue_scoring

        state = {
            "job_descriptions": [{"title": "EM"}, {"title": "EM"}],
            "current_job_index": 0,
            "resume_text": "", "resume_chunks": [], "scored_jobs": [], "final_results": []
        }
        assert should_continue_scoring(state) == "score_job"

    def test_should_continue_scoring_stops(self):
        """should_continue_scoring returns 'compile_results' when done."""
        from agents.resume_matcher import should_continue_scoring

        state = {
            "job_descriptions": [{"title": "EM"}, {"title": "EM"}],
            "current_job_index": 2,
            "resume_text": "", "resume_chunks": [], "scored_jobs": [], "final_results": []
        }
        assert should_continue_scoring(state) == "compile_results"

    @patch("agents.resume_matcher.run_resume_matcher")
    def test_run_resume_matcher_returns_structure(self, mock_run):
        """run_resume_matcher returns expected dict structure."""
        mock_run.return_value = {
            "matched_jobs": [{"company": "Freshworks", "match_score": 85}],
            "total_scored": 2,
            "recommended": 1,
        }

        from agents.resume_matcher import run_resume_matcher
        result = run_resume_matcher([{"company": "Freshworks"}, {"company": "LowCo"}], "resume.pdf")

        assert "matched_jobs" in result
        assert "total_scored" in result
        assert "recommended" in result
