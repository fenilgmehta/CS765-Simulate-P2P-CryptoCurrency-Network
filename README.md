# CS765-Simulate-P2P-CryptoCurrency-Network

- **Subject** - Introduction to Blockchains, Cryptocurrencies and Smart Contracts (CS 765)
- **Project** - Build a discrete-event Simulator for a P2P Cryptocurrency Network
    - [Problem Statement PDF](./CS765_Autum2021_HW1.pdf)
- **Team Members**
    - 203050054 - Fenil Mehta
    - 203059006 - Aditya Pradhan
    - 20305R005 - Arnab Das

### Points to Note

- `Node == Peer == Miner` all three mean the same
- `txn == transaction`
- `Unit of time = seconds` for everything
- `md5` hash function is used for speed - Can be updated to use any other hash function
- `f-strings` feature of Python is used
- `if` a block with `block_index < node.curr_block_index_max` is received, it is __dropped__ because:
    1. The received block is too old
    2. The block creator would not spend his time and computation power in creating and sending a block which does not
       create a longest chain for the receiver
- `mining reward` is always the first transaction of any block, and sender is `-1`
    - `Sender == -1` means money is created from thin air
- `mining_reward_update_percent` is used as follows: `new_reward = old_reward (1 + mining_reward_update_percent / 100)`
    - The above statement will be executed every `mining_reward_update_block_time` blocks
        - i.e. when `longest_chain.index % mining_reward_update_block_time == 0`

### Execution Steps

```shell
cd src
python simulator.py --config config.json --debug
```

### References

- https://treelib.readthedocs.io/en/latest/
- https://graphviz.org/Gallery/directed/fsm.html
- [Bitcoin and cryptocurrency mining explained](https://www.youtube.com/watch?v=kZXXDp0_R-w)
- [Proof-of-Stake (vs proof-of-work)](https://www.youtube.com/watch?v=M3EFi_POhps)
- [But how does bitcoin actually work?](https://www.youtube.com/watch?v=bBC-nXj3Ng4)
- Exponential Distribution
    - [https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html](https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html)
    - [https://en.wikipedia.org/wiki/Exponential_distribution](https://en.wikipedia.org/wiki/Exponential_distribution)
- [https://towardsdatascience.com/static-typing-in-python-55aa6dfe61b4](Static Typing in Python)
- [https://en.wikipedia.org/wiki/Byte#Multiple-byte_units](https://en.wikipedia.org/wiki/Byte#Multiple-byte_units)
    - KiloByte (kB) vs. KibiByte (KiB)
- [https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports](https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports)
