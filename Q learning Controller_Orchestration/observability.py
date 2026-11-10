"""Structured local event logging and lightweight trace correlation."""
import json
import os
from datetime import datetime, timezone
from uuid import uuid4


def new_trace_id():
    return uuid4().hex


class JsonlEventSink:
    """Append events to a local file; errors propagate so runs fail closed."""
    def __init__(self, path):
        self.path = os.fspath(path)

    def emit(self, trace_id, event, **fields):
        record = {'timestamp': datetime.now(timezone.utc).isoformat(),
                  'trace_id': trace_id, 'event_id': uuid4().hex,
                  'event': event, **fields}
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        flags |= getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(self.path, flags, 0o600)
        with os.fdopen(fd, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False, separators=(',', ':')) + '\n')
            stream.flush()
