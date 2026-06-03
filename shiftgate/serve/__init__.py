"""shiftgate serve — OpenAI-compatible HTTP proxy with auto-routing.

The public entry point is :func:`create_app`, which builds a FastAPI
application that exposes shiftgate's routing intelligence behind the standard
OpenAI ``/v1/chat/completions`` API.  Point any OpenAI client at it and pass
``model="auto"`` to get automatic LoRA adapter selection.
"""

from shiftgate.serve.app import create_app

__all__ = ["create_app"]
