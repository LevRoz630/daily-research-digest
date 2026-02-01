"""Tests for daily_research_digest.email_provider module."""

import smtplib
from unittest.mock import MagicMock, patch

import pytest

from daily_research_digest.config_env import SMTPConfig
from daily_research_digest.email_provider import (
    DryRunProvider,
    EmailSendError,
    SMTPProvider,
)


class TestSMTPProvider:
    """Tests for SMTPProvider class."""

    @pytest.fixture
    def smtp_config(self) -> SMTPConfig:
        """Return test SMTP config."""
        return SMTPConfig(
            host="smtp.example.com",
            port=587,
            user="testuser",
            password="testpass",
            use_tls=True,
        )

    @pytest.fixture
    def mock_smtp(self) -> MagicMock:
        """Return mock SMTP instance."""
        mock = MagicMock()
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        return mock

    def test_send_success(self, smtp_config: SMTPConfig) -> None:
        """Test successful email send."""
        with patch("smtplib.SMTP") as mock_smtp_class:
            mock_smtp = MagicMock()
            mock_smtp_class.return_value = mock_smtp

            provider = SMTPProvider(smtp_config)
            provider.send(
                subject="Test Subject",
                from_addr="from@example.com",
                to_addrs=["to@example.com"],
                text_body="Plain text",
                html_body="<p>HTML</p>",
            )

            # Verify SMTP was initialized correctly
            mock_smtp_class.assert_called_once_with("smtp.example.com", 587)

            # Verify TLS was started
            mock_smtp.starttls.assert_called_once()

            # Verify login
            mock_smtp.login.assert_called_once_with("testuser", "testpass")

            # Verify sendmail was called
            mock_smtp.sendmail.assert_called_once()
            call_args = mock_smtp.sendmail.call_args
            assert call_args[0][0] == "from@example.com"
            assert call_args[0][1] == ["to@example.com"]

            # Verify quit
            mock_smtp.quit.assert_called_once()

    def test_send_without_tls(self) -> None:
        """Test sending without TLS."""
        config = SMTPConfig(
            host="smtp.example.com",
            port=25,
            use_tls=False,
        )

        with patch("smtplib.SMTP") as mock_smtp_class:
            mock_smtp = MagicMock()
            mock_smtp_class.return_value = mock_smtp

            provider = SMTPProvider(config)
            provider.send(
                subject="Test",
                from_addr="from@example.com",
                to_addrs=["to@example.com"],
                text_body="text",
                html_body="<p>html</p>",
            )

            # TLS should not be started
            mock_smtp.starttls.assert_not_called()

    def test_send_without_auth(self) -> None:
        """Test sending without authentication."""
        config = SMTPConfig(
            host="smtp.example.com",
            port=25,
            user=None,
            password=None,
            use_tls=False,
        )

        with patch("smtplib.SMTP") as mock_smtp_class:
            mock_smtp = MagicMock()
            mock_smtp_class.return_value = mock_smtp

            provider = SMTPProvider(config)
            provider.send(
                subject="Test",
                from_addr="from@example.com",
                to_addrs=["to@example.com"],
                text_body="text",
                html_body="<p>html</p>",
            )

            # Login should not be called
            mock_smtp.login.assert_not_called()

    def test_send_multiple_recipients(self, smtp_config: SMTPConfig) -> None:
        """Test sending to multiple recipients."""
        with patch("smtplib.SMTP") as mock_smtp_class:
            mock_smtp = MagicMock()
            mock_smtp_class.return_value = mock_smtp

            provider = SMTPProvider(smtp_config)
            provider.send(
                subject="Test",
                from_addr="from@example.com",
                to_addrs=["a@example.com", "b@example.com", "c@example.com"],
                text_body="text",
                html_body="<p>html</p>",
            )

            call_args = mock_smtp.sendmail.call_args
            assert call_args[0][1] == ["a@example.com", "b@example.com", "c@example.com"]

    def test_send_auth_failure(self, smtp_config: SMTPConfig) -> None:
        """Test authentication failure raises EmailSendError."""
        with patch("smtplib.SMTP") as mock_smtp_class:
            mock_smtp = MagicMock()
            mock_smtp_class.return_value = mock_smtp
            mock_smtp.login.side_effect = smtplib.SMTPAuthenticationError(535, b"Auth failed")

            provider = SMTPProvider(smtp_config)
            with pytest.raises(EmailSendError, match="authentication failed"):
                provider.send(
                    subject="Test",
                    from_addr="from@example.com",
                    to_addrs=["to@example.com"],
                    text_body="text",
                    html_body="<p>html</p>",
                )

    def test_send_connection_failure(self, smtp_config: SMTPConfig) -> None:
        """Test connection failure raises EmailSendError."""
        with patch("smtplib.SMTP") as mock_smtp_class:
            mock_smtp_class.side_effect = smtplib.SMTPConnectError(421, b"Connection refused")

            provider = SMTPProvider(smtp_config)
            with pytest.raises(EmailSendError, match="Failed to connect"):
                provider.send(
                    subject="Test",
                    from_addr="from@example.com",
                    to_addrs=["to@example.com"],
                    text_body="text",
                    html_body="<p>html</p>",
                )

    def test_send_recipients_refused(self, smtp_config: SMTPConfig) -> None:
        """Test recipients refused raises EmailSendError."""
        with patch("smtplib.SMTP") as mock_smtp_class:
            mock_smtp = MagicMock()
            mock_smtp_class.return_value = mock_smtp
            mock_smtp.sendmail.side_effect = smtplib.SMTPRecipientsRefused(
                {"bad@example.com": (550, b"User unknown")}
            )

            provider = SMTPProvider(smtp_config)
            with pytest.raises(EmailSendError, match="Recipients refused"):
                provider.send(
                    subject="Test",
                    from_addr="from@example.com",
                    to_addrs=["bad@example.com"],
                    text_body="text",
                    html_body="<p>html</p>",
                )

    def test_send_network_error(self, smtp_config: SMTPConfig) -> None:
        """Test network error raises EmailSendError."""
        with patch("smtplib.SMTP") as mock_smtp_class:
            mock_smtp_class.side_effect = OSError("Network unreachable")

            provider = SMTPProvider(smtp_config)
            with pytest.raises(EmailSendError, match="Network error"):
                provider.send(
                    subject="Test",
                    from_addr="from@example.com",
                    to_addrs=["to@example.com"],
                    text_body="text",
                    html_body="<p>html</p>",
                )

    def test_message_format(self, smtp_config: SMTPConfig) -> None:
        """Test email message is formatted correctly."""
        with patch("smtplib.SMTP") as mock_smtp_class:
            mock_smtp = MagicMock()
            mock_smtp_class.return_value = mock_smtp

            provider = SMTPProvider(smtp_config)
            provider.send(
                subject="Test Subject",
                from_addr="from@example.com",
                to_addrs=["to@example.com"],
                text_body="Plain text body",
                html_body="<p>HTML body</p>",
            )

            # Get the message that was sent
            message_str = mock_smtp.sendmail.call_args[0][2]

            # Check headers are present
            assert "Subject: Test Subject" in message_str
            assert "From: from@example.com" in message_str
            assert "To: to@example.com" in message_str
            assert "Content-Type: multipart/alternative" in message_str

            # Check both body types are indicated (content may be base64 encoded)
            assert "text/plain" in message_str
            assert "text/html" in message_str


class TestDryRunProvider:
    """Tests for DryRunProvider class."""

    def test_records_email(self) -> None:
        """Test email is recorded without sending."""
        provider = DryRunProvider()

        provider.send(
            subject="Test Subject",
            from_addr="from@example.com",
            to_addrs=["to@example.com"],
            text_body="Plain text",
            html_body="<p>HTML</p>",
        )

        assert len(provider.sent_emails) == 1
        email = provider.sent_emails[0]
        assert email["subject"] == "Test Subject"
        assert email["from_addr"] == "from@example.com"
        assert email["to_addrs"] == ["to@example.com"]
        assert email["text_body"] == "Plain text"
        assert email["html_body"] == "<p>HTML</p>"

    def test_records_multiple_emails(self) -> None:
        """Test multiple emails are recorded."""
        provider = DryRunProvider()

        provider.send("Subject 1", "a@b.com", ["c@d.com"], "text1", "html1")
        provider.send("Subject 2", "e@f.com", ["g@h.com"], "text2", "html2")

        assert len(provider.sent_emails) == 2
        assert provider.sent_emails[0]["subject"] == "Subject 1"
        assert provider.sent_emails[1]["subject"] == "Subject 2"
