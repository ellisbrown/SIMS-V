import multiprocessing as mp

from sims.data_generation.queues import FromToQueue


def test_local_queue_round_trip():
    pending = mp.Queue()
    completed = mp.Queue()
    queue = FromToQueue(from_queue=pending, to_queue=completed)

    pending.put("split_val__house_00000001__repeats_01")
    message = queue.get(timeout=1)

    assert message.body == "split_val__house_00000001__repeats_01"
    assert message.message_id

    queue.mark_complete(message)
    assert completed.get(timeout=1) == message.body
