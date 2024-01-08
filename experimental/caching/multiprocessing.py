from typing import Iterator

import torch.multiprocessing as mp 
from transformers import AutoTokenizer, AutoModel
import torch
from queue import Empty
import time
from enum import Enum

class SignalMessages(Enum):
    POISON_KILL = 1
    WAKE_ON_NOTIFY = 2

class SpecialQueue:
    def __init__(self, *args, **kwargs):
        self.queue = mp.Queue(*args, **kwargs)
       
    def get(self, block: bool = True, timeout: float | None = None):
        item = self.queue.get(block, timeout)
        if isinstance(item, SignalMessages):
            if item == SignalMessages.POISON_KILL:
                self.queue.put(SignalMessages.POISON_KILL)
                raise ValueError("Poison Kill")
            elif item == SignalMessages.WAKE_ON_NOTIFY:
                return None
        else:
            return item
        
    def put(self, item, block: bool = True, timeout: float | None = None):
        self.queue.put(item, block, timeout)
        
    def close(self):
        self.queue.close()
        
    def notify(self):
        self.queue.put(SignalMessages.WAKE_ON_NOTIFY)
# SpecialQueue = Queue

def queuebatcher(queue_in: SpecialQueue, queue_out: SpecialQueue, batch_size: int, sort_n=4):
    queue_in = queue_in
    queue_out =queue_out
    batch_size = batch_size
    sort_n = sort_n
    waiting_list = []
    while True:
        new_items = queue_in.get()
        
        if new_items is not None and len(new_items) > 0:
            waiting_list.extend(new_items)

        if new_items is None and waiting_list:
            # flush without waiting for a full batch
            queue_out.put(waiting_list[:batch_size])
            waiting_list = waiting_list[batch_size:]
        elif len(waiting_list) >= batch_size:
            # pop up to sort_n batches
            max_pop = min(sort_n, len(waiting_list)//batch_size)
            to_add, waiting_list = waiting_list[:batch_size*max_pop], waiting_list[batch_size*max_pop:]
            # sort and append
            to_add = sorted(to_add)
            for bs in range(0, len(to_add), batch_size):
                queue_out.put(to_add[bs:bs+batch_size])


class BoringPipeline(object):
    def __init__(self):
        pass
        
    def working_function(self, item):
        raise NotImplementedError
    
    def post_init(self, **kwargs):
        raise NotImplementedError
    
    def post_init_and_loop(self, queue_in: SpecialQueue, queue_out: SpecialQueue, **kwargs):
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.post_init(**kwargs)
        self.loop_forever()        
    
    def loop_forever(self):
        try:
            while True:
                item = self.queue_in.get()
                processed = self.working_function(item)
                self.queue_out.put(processed)
        except KeyboardInterrupt:
            pass
    
class TokenizePipeline(BoringPipeline):
    def post_init(self):
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        
        
    def working_function(self, item):
        assert isinstance(item, list) and all(isinstance(i, str) for i in item)
        try:
            with torch.inference_mode():
                return self.tokenizer(item, padding="max_length", truncation=True, return_tensors="pt")
        except Exception as ex:
            print(ex)
            return None

class ModelPipeline(BoringPipeline):
    def post_init(self, model_device: str):
        self.model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(model_device)
        self.model.eval()
        self.model.half()
        
    def working_function(self, item):
        with torch.inference_mode():
            item = item.to(self.model.device)
            output = self.model(**item).last_hidden_state
            return output.detach().cpu().shape

def main():    
    mp.set_start_method('spawn')
    queues = [SpecialQueue(), SpecialQueue(), SpecialQueue(), SpecialQueue()]
    
    # fill with some data
    items = [f"{i}" for i in range(5000)]
    # go  
    
    processes = []  
    processes.append(mp.Process(target=queuebatcher, args=(queues[0], queues[1], 64)))
    processes[-1].start()

    processes.append(mp.Process(target=TokenizePipeline().post_init_and_loop, kwargs=dict(
        queue_in=queues[1], queue_out=queues[2], device="cuda")))
    processes[-1].start()  

    processes.append(mp.Process(target=ModelPipeline().post_init_and_loop, kwargs=dict(
        queue_in=queues[2], queue_out=queues[3], model_device="cuda")))
    processes[-1].start()
    queues[0].put(items[1:33]) 
    time.sleep(5)
    
    
    s = time.perf_counter()
    for bs in range(0, len(items), 17):
        queues[0].put(items[bs:bs+17]) 
    time.sleep(2)
    try:
        i = 0
        while i < 1:
            try:
                item = queues[-1].get(timeout=0.5)
            except Empty:
                queues[0].put(SignalMessages.WAKE_ON_NOTIFY)
                i+=1
                continue
            print(item)
    finally:
        print(time.perf_counter() -s, "seconds")
        print("Shutting down")
        for i in range(5):
            for q in queues:
                q.put(SignalMessages.POISON_KILL)
        time.sleep(3)
        print("closing queues")
        for queue in queues:
            queue.close()
        # print("joining processes")
        # for p in processes:
        #     p.join()
        queues = None
        # time.sleep(3)
        print("Done")
        
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    start = time.perf_counter()
    model.encode(items, batch_size=64, show_progress_bar=True)
    print(time.perf_counter() - start, "sentence transformers")
            
        
    
        
if __name__ == "__main__":
    main()