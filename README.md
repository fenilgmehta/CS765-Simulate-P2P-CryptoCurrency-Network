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
- `f-strings` feature of Python is used
- `md5` hash function is used for speed
  	- Can be updated to use any other hash function

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
- [https://towardsdatascience.com/static-typing-in-python-55aa6dfe61b4](Static Typing in Python)
- [https://en.wikipedia.org/wiki/Byte#Multiple-byte_units](https://en.wikipedia.org/wiki/Byte#Multiple-byte_units)
	- KiloByte (kB) vs. KibiByte (KiB)