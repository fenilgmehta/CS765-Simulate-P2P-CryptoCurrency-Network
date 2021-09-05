from transaction import Transaction
import hashlib
from typing import List

class Block:
    def __init__(self, prev_block_hash: str, creation_time: float, index: int, transactions: List[Transaction],
                 recv_time: float, mine_time: float = -1.0):
        """self.index is 0-indexed"""
        self.prev_block_hash: str = prev_block_hash  # Hash of the previous block
        self.creation_time: float = creation_time  # Time when the block was created and mining started
        self.index: int = index  # This is 0-indexed, and Genesis Block is at index 0.
        self.transactions: List[Transaction] = transactions  # List of all transactions

        # The below values/variables are NOT used during hash calculation of this block
        self.curr_block_hash: str = self.get_hash()  # Hash of the current block
        self.recv_time: float = recv_time  # Time when the block was received
        self.mine_time: float = mine_time  # Time when the this block was successfully mined

    def __str__(self) -> str:
        """
        NOTE: this does not include block hash
        String of "self.prev_block_hash", "self.creation_time", "self.index" and "self.transaction"
        """
        return str([self.prev_block_hash, self.creation_time, self.index, [str(i) for i in self.transactions]])

    def str_all(self) -> str:
        return str([self.curr_block_hash, self.recv_time,  # These both are not present in __str__
                    self.prev_block_hash, self.creation_time, self.index, [str(i) for i in self.transactions]])

    def get_hash(self) -> str:
        """
        Hash is calculated based on "self.prev_block_hash", "self.creation_time" and "self.transactions"
        """
        return hashlib.md5(str(self).encode()).hexdigest()

    def size(self) -> int:
        # REFER: https://stackoverflow.com/questions/14329794/get-size-in-bytes-needed-for-an-integer-in-python
        # In real life
        # return len(self.prev_block_hash) \
        #        + sys.getsizeof(self.creation_time) \
        #        + sys.getsizeof(self.index) \
        #        + (Transaction.size() * len(self.transactions))
        # In our simulator
        return Transaction.size() * len(self.transactions)