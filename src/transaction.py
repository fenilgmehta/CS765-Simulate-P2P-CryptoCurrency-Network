import hashlib

class Transaction:
    def __init__(self, txn_time: float, id_sender: int, id_receiver: int, coin_amount: float):
        self.txn_time: float = txn_time
        self.id_sender: int = id_sender
        self.id_receiver: int = id_receiver
        self.coin_amount: float = coin_amount
        # Note: this is not used much in this simulator. However, it is very useful in real life
        self.txn_hash: str = self.get_hash()

    def __str__(self) -> str:
        return str([self.txn_time, self.id_sender, self.id_receiver, self.coin_amount])

    def __eq__(self, other) -> bool:
        return type(self) == type(other) and self.txn_hash == other.txn_hash

    def __hash__(self):
        return int(self.get_hash(), base=16)

    def get_hash(self) -> str:
        return hashlib.md5(str(self).encode()).hexdigest()

    @staticmethod
    def size() -> int:
        """
        Returns size in Bytes
        NOTE: Size is assumed to be 1KB (According to the Problem Statement PDF)
        """
        return 1000