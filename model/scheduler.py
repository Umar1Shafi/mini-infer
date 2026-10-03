from enum import Enum

class RequestStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"

class Request:
    def __init__(self, request_id, prompt_token_ids, max_new_tokens, eos_token_id):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids       # list[int], the original prompt
        self.generated_token_ids = []                   # tokens produced so far
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id
        self.status = RequestStatus.WAITING
        self.page_table = None                           # assigned once it starts running
        self.num_prompt_tokens = len(prompt_token_ids)

    def current_length(self):
        """Total tokens so far: prompt + whatever's been generated."""
        return self.num_prompt_tokens + len(self.generated_token_ids)

    def is_prefill(self):
        """True if this request hasn't processed its prompt yet (first step)."""
        return len(self.generated_token_ids) == 0

    def next_input_ids(self):
        """
        What to feed the model this step:
        - prefill: the whole prompt
        - decode: just the last generated token
        """
        if self.is_prefill():
            return self.prompt_token_ids
        return [self.generated_token_ids[-1]]

    def append_token(self, token_id):
        self.generated_token_ids.append(token_id)
        if token_id == self.eos_token_id or len(self.generated_token_ids) >= self.max_new_tokens:
            self.status = RequestStatus.FINISHED

    def all_token_ids(self):
        return self.prompt_token_ids + self.generated_token_ids


class Scheduler:
    def __init__(self, max_batch_size):
        self.max_batch_size = max_batch_size
        self.waiting = []    # requests not yet admitted
        self.running = []    # requests currently being processed

    def add_request(self, request):
        self.waiting.append(request)

    def step_admit(self):
        """
        Admit waiting requests into the running batch, up to max_batch_size.
        Call this once per generation step, before running the model.
        Returns the list of newly admitted requests (for logging/testing).
        """
        newly_admitted = []
        while self.waiting and len(self.running) < self.max_batch_size:
            req = self.waiting.pop(0)
            req.status = RequestStatus.RUNNING
            self.running.append(req)
            newly_admitted.append(req)
        return newly_admitted

    def step_evict_finished(self):
        """
        Remove finished requests from the running batch.
        Call this once per generation step, after processing tokens.
        Returns the list of requests that finished this step (for logging/testing/freeing pages).
        """
        still_running = []
        finished = []
        for req in self.running:
            if req.status == RequestStatus.FINISHED:
                finished.append(req)
            else:
                still_running.append(req)
        self.running = still_running
        return finished

    def has_work(self):
        return len(self.waiting) > 0 or len(self.running) > 0
