"""Tests for daily_research_digest.config_env module."""

import os
from unittest.mock import patch

import pytest

from daily_research_digest.config_env import (
    ConfigError,
    load_config_from_env,
    parse_bool,
    parse_list,
    parse_window,
)


class TestParseWindow:
    """Tests for parse_window function."""

    def test_parse_hours(self) -> None:
        """Test parsing hour format."""
        delta = parse_window("24h")
        assert delta.total_seconds() == 24 * 3600

    def test_parse_hours_various(self) -> None:
        """Test various hour values."""
        assert parse_window("1h").total_seconds() == 3600
        assert parse_window("48h").total_seconds() == 48 * 3600
        assert parse_window("168h").total_seconds() == 168 * 3600

    def test_parse_days(self) -> None:
        """Test parsing day format."""
        delta = parse_window("1d")
        assert delta.total_seconds() == 24 * 3600

    def test_parse_days_various(self) -> None:
        """Test various day values."""
        assert parse_window("7d").total_seconds() == 7 * 24 * 3600
        assert parse_window("30d").total_seconds() == 30 * 24 * 3600

    def test_parse_case_insensitive(self) -> None:
        """Test case insensitive parsing."""
        assert parse_window("24H").total_seconds() == 24 * 3600
        assert parse_window("1D").total_seconds() == 24 * 3600

    def test_parse_with_whitespace(self) -> None:
        """Test parsing with surrounding whitespace."""
        assert parse_window("  24h  ").total_seconds() == 24 * 3600

    def test_parse_invalid_format(self) -> None:
        """Test invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid window format"):
            parse_window("24")

    def test_parse_invalid_suffix(self) -> None:
        """Test invalid suffix raises ValueError."""
        with pytest.raises(ValueError, match="Invalid window format"):
            parse_window("24x")

    def test_parse_invalid_number(self) -> None:
        """Test invalid number raises ValueError."""
        with pytest.raises(ValueError, match="Invalid window format"):
            parse_window("abch")


class TestParseBool:
    """Tests for parse_bool function."""

    def test_true_values(self) -> None:
        """Test values that should be True."""
        assert parse_bool("true") is True
        assert parse_bool("True") is True
        assert parse_bool("TRUE") is True
        assert parse_bool("1") is True
        assert parse_bool("yes") is True
        assert parse_bool("on") is True

    def test_false_values(self) -> None:
        """Test values that should be False."""
        assert parse_bool("false") is False
        assert parse_bool("False") is False
        assert parse_bool("0") is False
        assert parse_bool("no") is False
        assert parse_bool("off") is False
        assert parse_bool("") is False


class TestParseList:
    """Tests for parse_list function."""

    def test_simple_list(self) -> None:
        """Test simple comma-separated list."""
        result = parse_list("a,b,c")
        assert result == ["a", "b", "c"]

    def test_list_with_spaces(self) -> None:
        """Test list with spaces around items."""
        result = parse_list("a@b.com, c@d.com , e@f.com")
        assert result == ["a@b.com", "c@d.com", "e@f.com"]

    def test_empty_string(self) -> None:
        """Test empty string returns empty list."""
        assert parse_list("") == []
        assert parse_list("   ") == []

    def test_single_item(self) -> None:
        """Test single item list."""
        assert parse_list("single") == ["single"]

    def test_filters_empty_items(self) -> None:
        """Test empty items are filtered out."""
        result = parse_list("a,,b,  ,c")
        assert result == ["a", "b", "c"]


class TestLoadConfigFromEnv:
    """Tests for load_config_from_env function."""

    def _base_env(self) -> dict[str, str]:
        """Return base required environment variables."""
        return {
            "DIGEST_RECIPIENTS": "test@example.com",
            "DIGEST_CATEGORIES": "cs.AI,cs.LG",
            "DIGEST_INTERESTS": "machine learning research",
            "SMTP_HOST": "smtp.example.com",
            "ANTHROPIC_API_KEY": "test-key",
        }

    def test_load_with_required_vars(self) -> None:
        """Test loading with all required variables."""
        env = self._base_env()
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env()

        assert config.recipients == ["test@example.com"]
        assert config.categories == ["cs.AI", "cs.LG"]
        assert config.interests == "machine learning research"
        assert config.smtp_config is not None
        assert config.smtp_config.host == "smtp.example.com"

    def test_load_multiple_recipients(self) -> None:
        """Test parsing multiple recipients."""
        env = self._base_env()
        env["DIGEST_RECIPIENTS"] = "a@b.com, c@d.com, e@f.com"
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env()

        assert config.recipients == ["a@b.com", "c@d.com", "e@f.com"]

    def test_missing_recipients_raises(self) -> None:
        """Test missing recipients raises ConfigError."""
        env = self._base_env()
        del env["DIGEST_RECIPIENTS"]
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigError, match="DIGEST_RECIPIENTS"):
                load_config_from_env()

    def test_missing_categories_raises(self) -> None:
        """Test missing categories raises ConfigError."""
        env = self._base_env()
        del env["DIGEST_CATEGORIES"]
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigError, match="DIGEST_CATEGORIES"):
                load_config_from_env()

    def test_missing_interests_raises(self) -> None:
        """Test missing interests raises ConfigError."""
        env = self._base_env()
        del env["DIGEST_INTERESTS"]
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigError, match="DIGEST_INTERESTS"):
                load_config_from_env()

    def test_missing_smtp_host_raises(self) -> None:
        """Test missing SMTP host raises ConfigError."""
        env = self._base_env()
        del env["SMTP_HOST"]
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigError, match="SMTP_HOST"):
                load_config_from_env()

    def test_missing_api_key_raises(self) -> None:
        """Test missing API key for provider raises ConfigError."""
        env = self._base_env()
        del env["ANTHROPIC_API_KEY"]
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
                load_config_from_env()

    def test_defaults_applied(self) -> None:
        """Test default values are applied."""
        env = self._base_env()
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env()

        assert config.subject == "Daily Research Digest - {date}"
        assert config.from_addr == "noreply@example.com"
        assert config.timezone == "UTC"
        assert config.window == "24h"
        assert config.max_papers == 50
        assert config.top_n == 10
        assert config.llm_provider == "anthropic"
        assert config.smtp_config.port == 587
        assert config.smtp_config.use_tls is True

    def test_custom_values(self) -> None:
        """Test custom values override defaults."""
        env = self._base_env()
        env.update(
            {
                "DIGEST_SUBJECT": "Custom Subject - {date}",
                "DIGEST_FROM": "custom@example.com",
                "DIGEST_TZ": "US/Pacific",
                "DIGEST_WINDOW": "48h",
                "DIGEST_MAX_PAPERS": "100",
                "DIGEST_TOP_N": "20",
                "SMTP_PORT": "465",
                "SMTP_USER": "user",
                "SMTP_PASS": "pass",
                "SMTP_TLS": "false",
            }
        )
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env()

        assert config.subject == "Custom Subject - {date}"
        assert config.from_addr == "custom@example.com"
        assert config.timezone == "US/Pacific"
        assert config.window == "48h"
        assert config.max_papers == 100
        assert config.top_n == 20
        assert config.smtp_config.port == 465
        assert config.smtp_config.user == "user"
        assert config.smtp_config.password == "pass"
        assert config.smtp_config.use_tls is False

    def test_openai_provider(self) -> None:
        """Test OpenAI provider configuration."""
        env = self._base_env()
        del env["ANTHROPIC_API_KEY"]
        env["LLM_PROVIDER"] = "openai"
        env["OPENAI_API_KEY"] = "openai-key"
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env()

        assert config.llm_provider == "openai"
        assert config.openai_api_key == "openai-key"

    def test_google_provider(self) -> None:
        """Test Google provider configuration."""
        env = self._base_env()
        del env["ANTHROPIC_API_KEY"]
        env["LLM_PROVIDER"] = "google"
        env["GOOGLE_API_KEY"] = "google-key"
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env()

        assert config.llm_provider == "google"
        assert config.google_api_key == "google-key"

    def test_invalid_window_format_raises(self) -> None:
        """Test invalid window format raises ConfigError."""
        env = self._base_env()
        env["DIGEST_WINDOW"] = "invalid"
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigError, match="Invalid window format"):
                load_config_from_env()

    def test_s3_state_uri(self) -> None:
        """Test S3 state URI is captured."""
        env = self._base_env()
        env["DIGEST_STATE_S3_URI"] = "s3://bucket/state.json"
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env()

        assert config.state_s3_uri == "s3://bucket/state.json"
