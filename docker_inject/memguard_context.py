"""
MemGuard thread-local context: stores the current user message per agent_id
so the tool executor can validate writes against it.
Set before tool execution fires; cleared after step completes.
"""
import threading

_local = threading.local()


def set_user_message(agent_id: str, message: str) -> None:
    if not hasattr(_local, "msgs"):
        _local.msgs = {}
    _local.msgs[agent_id] = message


def get_user_message(agent_id: str) -> str:
    if not hasattr(_local, "msgs"):
        return ""
    return _local.msgs.get(agent_id, "")


def clear_user_message(agent_id: str) -> None:
    if hasattr(_local, "msgs"):
        _local.msgs.pop(agent_id, None)
