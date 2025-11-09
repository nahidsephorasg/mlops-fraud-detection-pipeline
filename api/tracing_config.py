"""
OpenTelemetry tracing configuration for distributed tracing.
"""

import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor


def setup_tracing(app, service_name: str):
    """
    Configure OpenTelemetry tracing with Tempo backend.

    Args:
        app: FastAPI application instance
        service_name: Name of the service for trace identification
    """
    tempo_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://tempo:4317")

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.namespace": "fraud-detection",
            "deployment.environment": os.getenv("ENVIRONMENT", "production"),
        }
    )

    tracer_provider = TracerProvider(resource=resource)

    otlp_exporter = OTLPSpanExporter(endpoint=tempo_endpoint, insecure=True)

    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)

    trace.set_tracer_provider(tracer_provider)

    FastAPIInstrumentor.instrument_app(app)

    RequestsInstrumentor().instrument()

    return trace.get_tracer(__name__)
