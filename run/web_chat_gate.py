"""Process-local replay/concurrency guard, not a replacement for approvals."""
import threading


class ChatRequestGate:
    def __init__(self, capacity=10000):
        self._lock = threading.Lock()
        self._seen = set()
        self._active = False
        self._capacity = capacity

    def begin(self, chat_id, message_id):
        if not all(isinstance(value, str) and value.strip() for value in (chat_id, message_id)):
            return "Chat and user message IDs are required"
        key = (chat_id, message_id)
        with self._lock:
            if key in self._seen:
                return "Message already received; send a new message to inspect current state"
            if self._active:
                return "Workflow is busy; wait for the current request to finish"
            if len(self._seen) >= self._capacity:
                return "Chat request history is full; restart the UI service after pending work finishes"
            self._seen.add(key)
            self._active = True
        return None

    def finish(self):
        with self._lock:
            self._active = False
