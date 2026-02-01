"""Email provider abstraction for digest sending."""

from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .config_env import SMTPConfig


class EmailSendError(Exception):
    """Raised when email sending fails."""

    pass


class EmailProvider(Protocol):
    """Protocol for email providers."""

    def send(
        self,
        subject: str,
        from_addr: str,
        to_addrs: list[str],
        text_body: str,
        html_body: str,
    ) -> None:
        """Send an email.

        Args:
            subject: Email subject line
            from_addr: Sender email address
            to_addrs: List of recipient email addresses
            text_body: Plain text body
            html_body: HTML body

        Raises:
            EmailSendError: If sending fails
        """
        ...


class SMTPProvider:
    """SMTP email provider using stdlib smtplib."""

    def __init__(self, config: SMTPConfig) -> None:
        """Initialize SMTP provider.

        Args:
            config: SMTP server configuration
        """
        self.config = config

    def send(
        self,
        subject: str,
        from_addr: str,
        to_addrs: list[str],
        text_body: str,
        html_body: str,
    ) -> None:
        """Send an email via SMTP.

        Args:
            subject: Email subject line
            from_addr: Sender email address
            to_addrs: List of recipient email addresses
            text_body: Plain text body
            html_body: HTML body

        Raises:
            EmailSendError: If sending fails
        """
        # Build multipart message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = ", ".join(to_addrs)

        # Attach plain text and HTML parts
        text_part = MIMEText(text_body, "plain", "utf-8")
        html_part = MIMEText(html_body, "html", "utf-8")

        # Attach plain text first, then HTML (email clients prefer last)
        msg.attach(text_part)
        msg.attach(html_part)

        try:
            if self.config.use_tls:
                server = smtplib.SMTP(self.config.host, self.config.port)
                server.starttls()
            else:
                server = smtplib.SMTP(self.config.host, self.config.port)

            try:
                # Authenticate if credentials provided
                if self.config.user and self.config.password:
                    server.login(self.config.user, self.config.password)

                # Send email
                server.sendmail(from_addr, to_addrs, msg.as_string())

            finally:
                server.quit()

        except smtplib.SMTPAuthenticationError as e:
            raise EmailSendError(f"SMTP authentication failed: {e}") from e
        except smtplib.SMTPConnectError as e:
            raise EmailSendError(f"Failed to connect to SMTP server: {e}") from e
        except smtplib.SMTPRecipientsRefused as e:
            raise EmailSendError(f"Recipients refused: {e}") from e
        except smtplib.SMTPException as e:
            raise EmailSendError(f"SMTP error: {e}") from e
        except OSError as e:
            raise EmailSendError(f"Network error: {e}") from e


class DryRunProvider:
    """Dry run provider that logs instead of sending.

    Useful for testing and debugging.
    """

    def __init__(self) -> None:
        """Initialize dry run provider."""
        self.sent_emails: list[dict] = []

    def send(
        self,
        subject: str,
        from_addr: str,
        to_addrs: list[str],
        text_body: str,
        html_body: str,
    ) -> None:
        """Record email details without sending.

        Args:
            subject: Email subject line
            from_addr: Sender email address
            to_addrs: List of recipient email addresses
            text_body: Plain text body
            html_body: HTML body
        """
        self.sent_emails.append(
            {
                "subject": subject,
                "from_addr": from_addr,
                "to_addrs": to_addrs,
                "text_body": text_body,
                "html_body": html_body,
            }
        )
