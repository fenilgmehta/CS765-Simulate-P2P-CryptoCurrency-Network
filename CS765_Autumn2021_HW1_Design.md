# CS765 Project

- Python

- InitialUserConfig.txt
	- Peers: n (range is `0` to `n-1`)
	- SlowNodePerdent: z
	- CpuPowerRandVariable: Tk (`0 <= k < n`)
	- Mean Time of Exponential distribution: T_tx
	- Create first transaction for each peer based on exponential distribution (point 2)
	- Connect the peers and make sure the network is connected (point 4)
	- GraphMatrix (Adjacency Matrix) of `n*n` and value = partially info useful to calculate latency at runtime (point 5)
		- `ρij + |m|/cij + dij`
			- `ρij` can be chosen from a uniform distribution between 10ms and 500ms at the start of the simulation.
			- |m| denotes the length of the message in bits
			- `cij` is the link speed in bits per second
				- 100 Mbps if both i and j are fast
				- 5 Mbps if either of the nodes is slow
			- `dij` is _random for each message_ - queuing delay at senders side (i.e. node i)
				- `dij` is randomly chosen from an exponential distribution with some mean `96kbits/ci,j`

- Peers Info
	- PeerId
	- SlowFast

- EventQueue
	- Event Structure: Time, EventType, FromWhichNode, ForWhichNode
	- Next Event time = currentTime + latency (point 5)

- Transaction
	- 1 KB
	- Format
		- Transaction ID: TxnID
		- Senders ID    : IDx
		- Receivers ID  : IDy
		- Coins: C
	- Verification/Validation
		- C <= Coins owned by IDx

1 7
2 2
3 4


time t
[2,4,7,10]
 ^
