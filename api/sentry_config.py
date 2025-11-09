"""
Sentry error monitoring configuration.
"""

import os
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration


def setup_sentry(service_name: str):
    """
    Configure Sentry for error monitoring and performance tracking.

    Args:
        service_name: Name of the service for error tracking
    """
    sentry_dsn = os.getenv("SENTRY_DSN", "")

    if not sentry_dsn:
        print("SENTRY_DSN not configured - Sentry error monitoring disabled")
        return

    environment = os.getenv("ENVIRONMENT", "production")

    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=environment,
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "1.0")),
        profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "1.0")),
        integrations=[
            FastApiIntegration(),
            StarletteIntegration(),
        ],
        release=os.getenv("APP_VERSION", "1.0.0"),
        server_name=service_name,
        attach_stacktrace=True,
        send_default_pii=False,
        before_send=before_send_hook,
    )

    print(f"Sentry initialized for {service_name} in {environment} environment")


def before_send_hook(event, hint):
    """
    Filter sensitive data before sending to Sentry.

    Args:
        event: Sentry event dictionary
        hint: Additional context

    Returns:
        Modified event or None to drop the event
    """
    if "exc_info" in hint:
        exc_type, exc_value, tb = hint["exc_info"]
        if isinstance(exc_value, Exception) and "404" in str(exc_value):
            return None

    event.setdefault("tags", {})
    event["tags"]["service_type"] = "mlops-microservice"

    return event
