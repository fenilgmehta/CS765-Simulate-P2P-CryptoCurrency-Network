import enum
import heapq
from typing import Union, List, Tuple
from transaction import Transaction
from block import Block

# REFER: https://www.tutorialspoint.com/enum-in-python
class EventType(enum.Enum):
    EVENT_UNDEFINED = 0
    EVENT_TRANSACTION_CREATE = 1  # Queue -> data_obj: Transaction
    EVENT_SEND_TRANSACTION = 2
    EVENT_RECV_TRANSACTION = 3  # Queue -> data_obj: Transaction
    EVENT_SEND_BLOCK = 4
    EVENT_RECV_BLOCK = 5  # Queue -> data_obj: Block
    EVENT_BLOCK_CREATE = 6
    EVENT_BLOCK_CREATE_SUCCESS = 7  # Queue -> data_obj: Block


class Event:
    def __init__(
            self,
            event_completion_time: float,
            event_type: EventType,
            event_creator_id: int,
            event_receiver_id: int,
            data_obj: Union[None, Transaction, Block]
    ):
        self.event_completion_time: float = event_completion_time
        self.event_type: EventType = event_type
        self.event_creator_id: int = event_creator_id
        self.event_receiver_id: int = event_receiver_id
        self.data_obj = data_obj

    def __str__(self) -> str:
        return f'Event({self.event_type.name}, {self.event_completion_time:.2f}, ' \
               f'{self.event_creator_id: 2d}, {self.event_receiver_id: 2d}, {self.data_obj})'

    def str_all(self) -> str:
        if type(self.data_obj) == Block:
            res = f'Event({self.event_type.name}, {self.event_completion_time:.2f}, ' \
                  f'{self.event_creator_id: 2d}, {self.event_receiver_id: 2d}, Block('
            block_str = str([self.data_obj.prev_block_hash, self.data_obj.creation_time,
                             self.data_obj.index, len(self.data_obj.transactions)])
            return res + block_str + '))'
        else:
            return f'Event({self.event_type.name}, {self.event_completion_time:.2f}, ' \
                   f'{self.event_creator_id: 2d}, {self.event_receiver_id: 2d}, {self.data_obj})'


# REFER: https://docs.python.org/3/library/heapq.html
# REFER: https://www.geeksforgeeks.org/heap-queue-or-heapq-in-python/
class EventQueue:
    def __init__(self):
        # Tuple -> EventCompletionTime, Event
        self.events: List[Tuple[float, Event]] = list()
        self.add_new_events: bool = True

    def push(self, new_event: Event) -> None:
        if self.add_new_events == False:
            return
        heapq.heappush(self.events, (new_event.event_completion_time, new_event))

    def pop(self) -> Event:
        return heapq.heappop(self.events)[1]

    def top(self) -> Event:
        return self.events[0][1]

    def empty(self) -> bool:
        return len(self.events) == 0

    def freeze(self) -> None:
        self.add_new_events = False