from model.scheduler import Scheduler, Request, RequestStatus

def make_request(rid, prompt_len=3, max_new=5, eos=999):
    return Request(request_id=rid, prompt_token_ids=list(range(prompt_len)), max_new_tokens=max_new, eos_token_id=eos)

def test_admission_respects_batch_size():
    sched = Scheduler(max_batch_size=2)
    sched.add_request(make_request("a"))
    sched.add_request(make_request("b"))
    sched.add_request(make_request("c"))

    admitted = sched.step_admit()
    assert len(admitted) == 2, f"Expected 2 admitted, got {len(admitted)}"
    assert len(sched.running) == 2
    assert len(sched.waiting) == 1
    assert sched.waiting[0].request_id == "c"
    print("test_admission_respects_batch_size: PASS")

def test_late_arrival_fills_freed_slot():
    """Core continuous batching behavior: a finished request's slot is immediately reusable."""
    sched = Scheduler(max_batch_size=2)
    sched.add_request(make_request("a"))
    sched.add_request(make_request("b"))
    sched.step_admit()
    assert len(sched.running) == 2

    sched.add_request(make_request("c"))  # arrives while a, b are running
    assert len(sched.running) == 2, "c should NOT be running yet, batch is full"

    # "a" finishes
    sched.running[0].status = RequestStatus.FINISHED
    finished = sched.step_evict_finished()
    assert len(finished) == 1 and finished[0].request_id == "a"
    assert len(sched.running) == 1

    admitted = sched.step_admit()
    assert len(admitted) == 1 and admitted[0].request_id == "c"
    assert len(sched.running) == 2
    print("test_late_arrival_fills_freed_slot: PASS")

def test_request_lifecycle_prefill_then_decode():
    req = make_request("x", prompt_len=3, max_new=3, eos=999)
    req.status = RequestStatus.RUNNING  # simulate: scheduler has already admitted this request

    assert req.is_prefill() is True
    assert req.next_input_ids() == [0, 1, 2]

    req.append_token(10)
    assert req.is_prefill() is False
    assert req.next_input_ids() == [10]
    assert req.status == RequestStatus.RUNNING, "append_token must not change status while under max_new_tokens"

    req.append_token(11)
    req.append_token(12)  # this is the 3rd generated token -> hits max_new_tokens
    assert req.status == RequestStatus.FINISHED, "append_token SHOULD set FINISHED once max_new_tokens is hit"
    print("test_request_lifecycle_prefill_then_decode: PASS")

def test_eos_stops_generation_early():
    req = make_request("x", prompt_len=3, max_new=10, eos=999)
    req.append_token(5)
    req.append_token(999)  # EOS token, well before max_new_tokens=10
    assert req.status == RequestStatus.FINISHED
    assert len(req.generated_token_ids) == 2, "Should stop immediately at EOS, not pad to max_new_tokens"
    print("test_eos_stops_generation_early: PASS")

def test_has_work_reflects_true_state():
    sched = Scheduler(max_batch_size=2)
    assert sched.has_work() is False

    sched.add_request(make_request("a"))
    assert sched.has_work() is True

    sched.step_admit()
    assert sched.has_work() is True  # now running, still work

    sched.running[0].status = RequestStatus.FINISHED
    sched.step_evict_finished()
    assert sched.has_work() is False
    print("test_has_work_reflects_true_state: PASS")

if __name__ == "__main__":
    test_admission_respects_batch_size()
    test_late_arrival_fills_freed_slot()
    test_request_lifecycle_prefill_then_decode()
    test_eos_stops_generation_early()
    test_has_work_reflects_true_state()
    print("\nAll scheduler tests passed.")
