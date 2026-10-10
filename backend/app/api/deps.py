"""Shared route dependencies.

A session here is only a way to keep one browser tab's active dataset, active
model and workflow stage apart from another's. It carries no identity, no
credentials and no personal data, and it is never persisted beyond the running
process — the client generates the id and the server uses it as a dictionary
key. Requests that send no id fall into a shared "default" session, which is
what a single local user or a curl test gets.
"""

from fastapi import Header


def get_session_id(x_session_id: str = Header(default="")) -> str:
    cleaned = (x_session_id or "").strip()[:64]
    return cleaned or "default"
