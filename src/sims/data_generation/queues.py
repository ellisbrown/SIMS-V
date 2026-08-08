import uuid
from typing import Optional

from torch import multiprocessing as mp


class QueueMessage:
    """A work item exchanged by the local multiprocessing launcher."""

    def __init__(self, body: str):
        self.body = body
        self.message_id = str(uuid.uuid4())

    def __repr__(self) -> str:
        return f"QueueMessage({self.message_id}, {self.body})"


class FromToQueue:
    """Adapt a pair of multiprocessing queues to the worker interface."""

    def __init__(self, from_queue: mp.Queue, to_queue: mp.Queue):
        self.from_queue = from_queue
        self.to_queue = to_queue

    def get(self, timeout: Optional[float] = None) -> QueueMessage:
        return QueueMessage(body=self.from_queue.get(timeout=timeout))

    def mark_complete(self, message: QueueMessage) -> None:
        self.to_queue.put(message.body)
