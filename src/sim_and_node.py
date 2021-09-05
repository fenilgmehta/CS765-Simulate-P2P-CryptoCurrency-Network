import numpy
import random
import math
import copy
import traceback
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
from block import Block
from transaction import Transaction
from event import EventType
from event import Event
from event import EventQueue
from typing import Dict, List, Tuple, Union, Set
from collections import defaultdict
import parameters
from parameters import SimulatorParameters

def defaultdict_float():
    return defaultdict(float)

class Simulator:
    def __init__(self, sp: SimulatorParameters):
        self.simulator_parameters: SimulatorParameters = sp
        self.global_time: float = 0.0
        self.nodes_list: List[Node] = list()
        self.event_queue: EventQueue = EventQueue()
        self.freeze_everything_except_network: bool = False
        pass

    def initialize(self) -> None:
        # GENESIS_BLOCK = self.__create_genesis_block()
        GENESIS_BLOCK = self.__create_genesis_block_v2_empty()
        list_slow_fast: List[bool] = ([False] * self.simulator_parameters.number_of_slow_nodes) \
                                     + ([True] * self.simulator_parameters.number_of_fast_nodes)
        numpy.random.shuffle(list_slow_fast)

        list_malicious: List[bool] = [
            i < self.simulator_parameters.number_of_malicious_nodes
            for i in range(self.simulator_parameters.n_total_nodes)
        ]
        numpy.random.shuffle(list_malicious)
        debug_malicious_idx: List[int] = list()
        for i in range(self.simulator_parameters.n_total_nodes):
            if list_malicious[i]:
                debug_malicious_idx.append(i)
        parameters.g_logger.debug(f'{debug_malicious_idx=}')

        # Point 7 of PDF: Randomly generate CPU power of the nodes
        hash_power_percent: List[float] = list(numpy.random.random(self.simulator_parameters.n_total_nodes))
        hash_power_percent_sum = 100 / sum(hash_power_percent)
        hash_power_percent = [i * hash_power_percent_sum for i in hash_power_percent]
        parameters.g_logger.debug(f'{hash_power_percent=}')

        # Create the nodes
        self.nodes_list = [
            Node(i, self, i_hash_power, i_slow_fast, i_malicious, GENESIS_BLOCK)
            for i, (i_hash_power, i_slow_fast, i_malicious) in
            enumerate(zip(hash_power_percent, list_slow_fast, list_malicious))
        ]

        # Create a connected graph
        # Point 4 of the problem statement PDF
        self.__create_connected_graph()

        # Begin the infinite random transaction creation process
        # While handling an old transaction, a new transaction is created
        for node in self.nodes_list:
            node.transaction_create()
            node.mining_start()
        pass

    # REFER: https://www.geeksforgeeks.org/private-methods-in-python/
    def __create_genesis_block_v1(self) -> 'Block':
        """This method shall only be called during the start of the simulation"""
        min_of_block_limit_and_nodes = min(self.simulator_parameters.n_total_nodes,
                                           self.simulator_parameters.max_transactions_per_block)

        # REFER: https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html#numpy.random.randint
        # “discrete uniform” distribution

        # Uniform random distribution to nodes
        recv_node_idx = numpy.random.randint(
            low=0,
            high=self.simulator_parameters.n_total_nodes,
            size=min_of_block_limit_and_nodes
        )

        # Uniform random distribution of money
        # In real life, these coins will be with people instead of nodes
        # TODO: change to numpy.random.uniform to generate float value for initial coins a node has
        money = numpy.random.randint(
            low=self.simulator_parameters.node_initial_coins_low,
            high=self.simulator_parameters.node_initial_coins_high + 1,
            size=min_of_block_limit_and_nodes
        )

        # Sender = -1 denotes that coins are created from thin air in the genesis block
        transactions = [Transaction(0.0, -1, recv_idx, coins) for recv_idx, coins in zip(recv_node_idx, money)]
        return Block('-1', 0.0, 0, transactions, 0.0, 0.0)

    def __create_genesis_block_v2_empty(self) -> 'Block':
        """This method shall only be called during the start of the simulation"""
        return Block('-1', 0.0, 0, list(), 0.0, 0.0)

    def __create_connected_graph(self) -> None:
        """Point 4 of the PDF: create a connected network of nodes"""
        # REFER: https://www.scitepress.org/Papers/2014/49373/49373.pdf  
        # REFER: https://networkx.org/documentation/networkx-1.9.1/reference/generated/networkx.generators.random_graphs.barabasi_albert_graph.html
        # REFER: https://www.geeksforgeeks.org/barabasi-albert-graph-scale-free-models/
        # REFER: https://stackoverflow.com/questions/2041517/random-simple-connected-graph-generation-with-given-sparseness
        # REFER: https://stackoverflow.com/questions/6667201/how-to-define-a-two-dimensional-array

        # Total nodes in the graph
        n: int = self.simulator_parameters.n_total_nodes

        # Creating the barabasi albert graph for 'n' nodes
        g: nx.classes.graph.Graph = nx.barabasi_albert_graph(
            n,
            random.randint(1, 10 if n > 10 else max(1, n - 1))
        )

        # Point 5 of the PDF - latency time between sender "i" and receiver "j" for a message "m"
        #   ρ_ij is a positive minimum value corresponding to the speed of light propagation delay
        #   ρ_ij time is stored in "seconds"

        # The adjacency matrix (adj_mat) stores ρ_ij, and the matrix is NOT symmetric
        adj_mat: List[List[int]] = [[0 for i in range(n)] for j in range(n)]
        for i in range(n):
            for j in range(n):
                # Point 5 of the PDF
                adj_mat[i][j] = 0.010 + numpy.random.random() * (0.500 - 0.010)

        for edge in g.edges():
            node_i_id: int = edge[0]
            node_j_id: int = edge[1]

            # c_ij is the link speed between i and j
            # c_ij is in "bits per second"

            # If any of the nodes is SLOW
            c_ij = 5 * 1_000_000  # 5 Mbps

            if self.nodes_list[node_i_id].is_network_fast and self.nodes_list[node_j_id].is_network_fast:
                # If both of the nodes are FAST
                c_ij = 100 * 1_000_000  # 100 Mbps

            # Add node 'j' to the peers list of node 'i'
            self.nodes_list[node_i_id].add_new_peer(node_j_id, adj_mat[node_i_id][node_j_id], c_ij)
            # Add node 'i' to the peers list of node 'j'
            self.nodes_list[node_j_id].add_new_peer(node_i_id, adj_mat[node_j_id][node_i_id], c_ij)

        nx.draw(g, with_labels=True)
        plt.show()

    def freeze(self) -> None:
        self.freeze_everything_except_network = True
        # self.event_queue.freeze()

    def execute_next_event(self, execute_all_same_time_events: bool = True) -> bool:
        """This will execute all events with event_completion_time==queue.top().event_completion_time"""
        if self.event_queue.empty():
            parameters.g_logger.debug(f'Event queue is empty, returning...')
            return False

        event = self.event_queue.pop()
        while True:
            self.global_time = event.event_completion_time
            # NOTE: the "if" condition is used to reduce the number of redundant log operations
            if self.freeze_everything_except_network:
                if event.event_type in [EventType.EVENT_TRANSACTION_CREATE, EventType.EVENT_BLOCK_CREATE_SUCCESS]:
                    parameters.g_logger.debug(f'FREEZED: Time={self.get_global_time():.5f} , Event = {event.str_all()}')
                    break
                elif event.event_type not in [EventType.EVENT_TRANSACTION_CREATE, EventType.EVENT_RECV_TRANSACTION]:
                    parameters.g_logger.debug(f'Network: Time={self.get_global_time():.5f} , Event = {event.str_all()}')
            elif event.event_type not in [EventType.EVENT_TRANSACTION_CREATE, EventType.EVENT_RECV_TRANSACTION]:
                # NOTE: EventType.EVENT_RECV_TRANSACTION are not logged because they create a
                #       lot of log statements (as they are the most highly performed operations)
                #       and are of no use during debugging
                parameters.g_logger.debug(f'Time={self.get_global_time():.5f} , Event = {event.str_all()}')

            if event.event_type == EventType.EVENT_TRANSACTION_CREATE:
                txn: Transaction = event.data_obj
                self.nodes_list[event.event_receiver_id].transaction_event_handler(txn)
            elif event.event_type == EventType.EVENT_RECV_TRANSACTION:
                txn: Transaction = event.data_obj
                self.nodes_list[event.event_receiver_id].transaction_recv(txn, event.event_creator_id)
            elif event.event_type == EventType.EVENT_RECV_BLOCK:
                blk: Block = event.data_obj
                self.nodes_list[event.event_receiver_id].block_recv(blk, event.event_creator_id, self.global_time)
            elif event.event_type == EventType.EVENT_BLOCK_CREATE_SUCCESS:
                blk: Block = event.data_obj
                self.nodes_list[event.event_receiver_id].mining_complete(blk)
            else:
                parameters.g_logger.error(f'Problem: Unexpected EventType={event.event_type} , {event=}')

            if execute_all_same_time_events == False or \
                    self.event_queue.empty() or \
                    self.event_queue.top().event_completion_time > self.global_time:
                break
            event = self.event_queue.pop()
        return True

    def get_global_time(self) -> float:
        return self.global_time

    def write_all_node_tree_to_file(self):
        for node in self.nodes_list:
            self.write_node_tree_to_file(
                node_obj=node,
                file_name=f'{self.simulator_parameters.output_path}/tree_{node.node_id:03d}'
            )
        pass

    def write_node_tree_to_file(self, node_obj: 'Node', file_name: str):
        temp_name = file_name
        i = 0
        while os.path.exists(temp_name):
            parameters.g_logger.error(f'File already exists: "{file_name}"')
            temp_name = file_name + f'_({i:03d}).txt'
            i += 1
        with open(temp_name, 'w+') as f:
            f.write(node_obj.serialize_blockchain_to_str_v1())
            parameters.g_logger.info(f'Successfully written to file_name="{file_name}"')
        pass

class Node:
    """Structure of a node on the P2P network"""

    def __init__(self, node_id: int, simulator: Simulator, hash_power_percent: float, is_network_fast: bool,
                 is_malicious: bool, GENESIS_BLOCK: Block):
        self.node_id = node_id
        self.simulator: Simulator = simulator
        self.is_network_fast = is_network_fast
        self.is_malicious: bool = is_malicious

        # Dictionary of connected peers
        self.neighbors: Dict[int, Node.NodeSiblingInfo] = dict()
        # The time at which a new block with chain length > local chain length is received
        self.last_receive_time: float = -1.0

        self.txn_all: Dict[str, Transaction] = dict()  # Hash -> Transaction
        self.txn_pool: List[Transaction] = list()

        self.GENESIS_BLOCK = GENESIS_BLOCK
        self.blocks_all: Dict[str, Block] = {GENESIS_BLOCK.curr_block_hash: GENESIS_BLOCK}  # Hash -> Block
        self.blocks_unvalidated: Dict[str, Tuple[int, Block]] = dict()  # Hash -> (Sender, Block)
        self.blockchain_leafs: List[str] = [GENESIS_BLOCK.curr_block_hash]  # NOTE: this is always sorted

        self.cache_balance: Dict[str, Dict] = defaultdict(defaultdict_float)

        sp: SimulatorParameters = simulator.simulator_parameters

        # Point 7 of PDF: Exponential Distribution Mean for the
        # block mining time by node.
        t_meanTk = 1 / sp.T_k_block_avg_mining_time_sec
        t_lambda = hash_power_percent * t_meanTk / 100
        self.T_k_exp_block_mining_mean = numpy.random.exponential() / t_lambda

        # This is same for all nodes
        # Point 2 of PDF: Exponential Distribution Mean for inter-arrival time between transaction
        self.T_tx_exp_txn_interarrival_mean_sec = sp.T_tx_exp_txn_interarrival_mean_sec

        # This is same for all nodes
        # Max transactions a block can store
        self.max_transactions_per_block = sp.max_transactions_per_block

    def add_new_peer(self, new_peer_id: int, rho_ij: float, c_ij: int) -> bool:
        """
        Adding new peer to the neighbors list
        Inserts an edge between this and new_peer in the node graph
        """
        if new_peer_id in self.neighbors.keys():
            return False
        self.neighbors[new_peer_id] = Node.NodeSiblingInfo(new_peer_id, rho_ij, c_ij)
        return True

    def remove_peer(self, peer_id: int) -> bool:
        if peer_id not in self.neighbors.keys():
            return False
        self.neighbors.pop(peer_id)
        return True

    # REFER: https://www.geeksforgeeks.org/inner-class-in-python/
    class NodeSiblingInfo:
        def __init__(self, node_id: int, rho_ij: float, c_ij: int):
            self.node_id: int = node_id

            # Network latency parameters
            # ρ_ij = positive minimum value corresponding to speed of light propagation delay
            self.rho_ij: float = rho_ij
            # link speed between i and j in bits per second
            self.c_ij: int = c_ij

        def find_message_latency(self, message_size_bits: int) -> float:
            # Point 5 of the PDF
            #   - dij is the queuing delay at senders side (i.e. node i)
            #   - dij is randomly chosen from an exponential distribution with some mean `96kbits/c_ij`
            #   - NOTE: d_ij must be randomly chosen for each message transmitted from "i" to "j"
            d_ij = numpy.random.exponential(96_000 / self.c_ij)
            return self.rho_ij + (message_size_bits / self.c_ij) + d_ij

    def cache_update(self, new_tail: str) -> None:
        if new_tail in self.cache_balance.keys():
            return
        balance: Dict[int, float] = defaultdict(float)
        prev_block_hash: str = self.blocks_all[new_tail].prev_block_hash
        if prev_block_hash in self.cache_balance.keys():
            balance = self.cache_balance[prev_block_hash]
            self.cache_balance.pop(prev_block_hash)
            for txn in self.blocks_all[new_tail].transactions:
                balance[txn.id_sender] -= txn.coin_amount  # No need to handle Mining reward transaction
                balance[txn.id_receiver] += txn.coin_amount
        else:
            # self.cache_balance.clear()
            temp_tail = new_tail
            while temp_tail != self.GENESIS_BLOCK.prev_block_hash:
                for txn in self.blocks_all[temp_tail].transactions:
                    balance[txn.id_sender] -= txn.coin_amount
                    balance[txn.id_receiver] += txn.coin_amount
                if temp_tail in self.cache_balance.keys():
                    self.cache_balance.pop(temp_tail)
                temp_tail = self.blocks_all[temp_tail].prev_block_hash
        self.cache_balance[new_tail] = balance
        pass

    def change_mining_branch(self, block_tail_hash_new: str) -> None:
        block_tail_hash_old = self.blockchain_leafs[-1]
        # if block_tail_hash_old == self.blocks_all[block_tail_hash_new].prev_block_hash:
        #     for txn in self.blocks_all[block_tail_hash_new].transactions:
        #         self.txn_pool.remove(txn)
        #     return

        old_idx = self.blocks_all[block_tail_hash_old].index
        new_idx = self.blocks_all[block_tail_hash_new].index
        min_idx = min(old_idx, new_idx)

        ancestor_hash_old: str = block_tail_hash_old
        ancestor_hash_new: str = block_tail_hash_new
        while min_idx < self.blocks_all[ancestor_hash_old].index:
            ancestor_hash_old = self.blocks_all[ancestor_hash_old].prev_block_hash
        while min_idx < self.blocks_all[ancestor_hash_new].index:
            ancestor_hash_new = self.blocks_all[ancestor_hash_new].prev_block_hash
        while ancestor_hash_old != ancestor_hash_new:
            ancestor_hash_old = self.blocks_all[ancestor_hash_old].prev_block_hash
            ancestor_hash_new = self.blocks_all[ancestor_hash_new].prev_block_hash
        # Now, "ancestor_hash_old" and "ancestor_hash_new" point to the common ancestor

        # Add transactions of old branch to the transaction pool
        while ancestor_hash_old != block_tail_hash_old:
            # Be careful, First txn is Mining Reward
            if self.blocks_all[block_tail_hash_old].transactions[0].id_sender == -1:
                self.txn_pool.extend(self.blocks_all[block_tail_hash_old].transactions[1:])
            else:
                self.txn_pool.extend(self.blocks_all[block_tail_hash_old].transactions)
            block_tail_hash_old = self.blocks_all[block_tail_hash_old].prev_block_hash
        # Remove included transactions from the transaction pool
        while ancestor_hash_new != block_tail_hash_new:
            for txn in self.blocks_all[block_tail_hash_new].transactions:
                if txn.id_sender == -1:
                    continue  # Mining reward transactions are never put in transaction pool
                # NOTE: there is a possibility that a transaction may not have reached me
                #       but someone must have included it in the blockchain and I must have
                #       received the block even before I see the transaction
                try:
                    self.txn_pool.remove(txn)
                except ValueError:
                    parameters.g_logger.warning(f'Block received with a transaction which is not yet received, '
                                     f'self.node_id = {self.node_id} , '
                                     f'block_hash = {block_tail_hash_new} , txn.txn_hash = {txn.txn_hash}')
            block_tail_hash_new = self.blocks_all[block_tail_hash_new].prev_block_hash
        pass

    def is_transaction_valid(
            self,
            transaction_obj: Transaction,
            curr_tail: Union[str, None] = None,
            senders_balance_init: float = 0.0
    ) -> bool:

        if transaction_obj.txn_hash != transaction_obj.get_hash():
            return False

        if transaction_obj.coin_amount < 0.0:
            return False

        if transaction_obj.id_sender < 0:
            return False

        curr_blockchain_hash: Union[str, None] = curr_tail
        if curr_blockchain_hash is None:
            curr_blockchain_hash = self.blockchain_leafs[-1]

        senders_balance: float = senders_balance_init
        while curr_blockchain_hash != self.GENESIS_BLOCK.prev_block_hash:
            for txn in self.blocks_all[curr_blockchain_hash].transactions:
                if transaction_obj.txn_hash == txn.txn_hash:
                    # Transaction is already included in the past block
                    return False
                if transaction_obj.id_sender == txn.id_sender:
                    senders_balance -= txn.coin_amount
                elif transaction_obj.id_sender == txn.id_receiver:
                    senders_balance += txn.coin_amount
            curr_blockchain_hash = self.blocks_all[curr_blockchain_hash].prev_block_hash
        if senders_balance < 0.0:
            parameters.g_logger.error(f'-ve balance ---> node_id={self.node_id}, senders_balance = {senders_balance}, '
                           f'Blockchain tail = {self.blockchain_leafs[-1]}, transaction = {transaction_obj}')
            # TODO: do more verbose logging
        return senders_balance >= transaction_obj.coin_amount

    def get_balance(self, node_id, tail_block) -> float:
        # parameters.g_logger.debug(f'{self.blockchain_leafs=}')
        # parameters.g_logger.debug(f'{node_id=} , {tail_block=}')
        balance: float = 0.0
        while tail_block != '-1':
            for txn in self.blocks_all[tail_block].transactions:
                if node_id == txn.id_sender:
                    balance -= txn.coin_amount
                if node_id == txn.id_receiver:
                    balance += txn.coin_amount
            tail_block = self.blocks_all[tail_block].prev_block_hash
        return balance

    def transaction_create(self):
        """
        Point 2 of the PDF: Generate Transaction and add it to the event queue
          - Randomly Select a node for receiver
          - Coin Amount : generate randomly based on current balance
        """
        txn_receiver: int = random.choice(
            [node.node_id for node in self.simulator.nodes_list if node.node_id != self.node_id]
        )
        txn_amount: float = round(random.uniform(0, self.get_balance(self.node_id, self.blockchain_leafs[-1])), 2)
        txn_obj: Transaction = Transaction(
            self.simulator.get_global_time(),
            self.node_id,
            txn_receiver,
            txn_amount
        )

        next_txn_gen_event_delay = numpy.random.exponential(
            self.T_tx_exp_txn_interarrival_mean_sec
        )
        self.simulator.event_queue.push(
            Event(
                self.simulator.get_global_time() + next_txn_gen_event_delay,
                EventType.EVENT_TRANSACTION_CREATE,
                self.node_id,
                self.node_id,
                txn_obj
            )
        )

    def transaction_event_handler(self, txn_obj: Transaction):
        if txn_obj.txn_hash in self.txn_all.keys():
            parameters.g_logger.warning(f'Problem: Transaction generated with hash same as previous txn, {txn_obj=}')
        else:
            # if self.is_transaction_valid(txn_obj):
            #     self.txn_all[txn_obj.txn_hash] = txn_obj
            #     self.txn_pool.append(txn_obj)
            # # NOTE: even invalid transactions are sent to neighbors to prove that other
            # #       peers are working properly and they will discard invalid received data
            pass
            # NOTE: txn is valid/invalid only when it is put in a block because balance
            #       of the sender can change before it is included in any block
            self.txn_all[txn_obj.txn_hash] = txn_obj
            self.txn_pool.append(txn_obj)
            for node in self.neighbors.values():
                self.transaction_send(txn_obj, node)
        self.transaction_create()

    def transaction_send(self, transaction_obj: Transaction, receiver_node: NodeSiblingInfo):
        """
            Receiver Obj : Peer Node of type NodeSiblingInfo
            it computes the latency to send the transaction
            it then generates an event in the event queue at that time.
        """
        self.simulator.event_queue.push(
            Event(
                self.simulator.get_global_time() + receiver_node.find_message_latency(8 * transaction_obj.size()),
                EventType.EVENT_RECV_TRANSACTION,
                self.node_id,
                receiver_node.node_id,
                transaction_obj
            )
        )

    def transaction_recv(self, transaction_obj: Transaction, senders_id: int):
        # IF I have already received the transaction; THEN I drop it
        if transaction_obj.txn_hash in self.txn_all:
            return
        # NOTE: transaction should only be validated when putting in a block
        # # IF transaction is invalid; then I drop it
        # if self.is_transaction_valid(transaction_obj) == False:
        #     return
        if transaction_obj.id_sender <= -1 or transaction_obj.coin_amount < 0:
            return
        # Store the transaction, add it to txn pool and forward it to others
        self.txn_all[transaction_obj.txn_hash] = transaction_obj
        self.txn_pool.append(transaction_obj)
        for node in self.neighbors.values():
            # Do NOT send the transaction back to the node from which it was received
            if node.node_id == senders_id:
                continue
            self.transaction_send(transaction_obj, node)
        pass

    def is_block_valid(self, block_obj: Block) -> int:
        """
        NOTE: first transaction of all blocks SHOULD ONLY be mining reward transaction
        :returns -1 to denote "False", 0 to denote "can not say", 1 to denote "True"
        """

        # NOTE: Blocks with no transactions are allowed/valid
        # if len(block_obj.transactions) <= 1:
        #     return -1

        # Check if block content was modified or not
        if block_obj.get_hash() != block_obj.curr_block_hash:
            return -1

        if block_obj.prev_block_hash not in self.blocks_all:
            parameters.g_logger.warning(f'Block received whose parent is not yet received to this Node, '
                             f'time={self.simulator.get_global_time()}, '
                             f'node_id={self.node_id}, blk_hash={block_obj.curr_block_hash}')
            # parameters.g_logger.warning(f'{self.node_id=} , {block_obj.str_all()=}')
            # parameters.g_logger.debug(f'self.blocks_all.keys()   = {list(self.blocks_all.keys())}')
            # parameters.g_logger.debug(f'self.blocks_all.values() = {[i.str_all() for i in self.blocks_all.values()]}')
            # for node in self.simulator.nodes_list:
            #     if node.blocks_all[node.blockchain_leafs[-1]].index >= block_obj.index:
            #         parameters.g_logger.debug(f'\tnode.node_id = {node.node_id}')
            #         parameters.g_logger.debug(f'\tnode.blocks_all.keys()   = {[str(i) for i in node.blocks_all.keys()]}')
            #         parameters.g_logger.debug(f'\tnode.blocks_all.values() = {[i.str_all() for i in node.blocks_all.values()]}')
            return 0

        # Check creation time of the received block
        if block_obj.creation_time >= (
                self.simulator.get_global_time()
                + self.simulator.simulator_parameters.max_block_creation_delay_sec
        ):
            parameters.g_logger.warning(f'Problem: Block: Block Creation Time > currTime+max_block_creation_delay_sec')
            parameters.g_logger.debug(f'{block_obj.str_all()=}')
            return -1

        # Index of received block is invalid
        if self.blocks_all[block_obj.prev_block_hash].index + 1 != block_obj.index:
            parameters.g_logger.debug(f'Block: Invalid index block_obj={block_obj}, '
                           f'parent={self.blocks_all[block_obj.prev_block_hash]}')
            return -1

        # Miner may or may not want to take any reward
        if block_obj.transactions[0].id_sender < -1:
            parameters.g_logger.debug(f'Block: Invalid Mining Reward sender_id={block_obj.transactions[0]}')
            return -1  # Only -1 can be used for senders field in a Mining Reward transaction
        if block_obj.transactions[0].id_sender == -1:
            # Ensure mining fee is right
            c = block_obj.transactions[0].coin_amount
            sp = self.simulator.simulator_parameters
            c_expected = sp.mining_reward_start * \
                         (
                                 (1 + (sp.mining_reward_update_percent / 100)) **
                                 (block_obj.index // sp.mining_reward_update_block_time)
                         )
            # REFER: https://stackoverflow.com/questions/5595425/what-is-the-best-way-to-compare-floats-for-almost-equality-in-python
            if not math.isclose(c, c_expected):
                parameters.g_logger.warning(f'Problem: Block: Invalid transaction fee, Expected={c_expected}, Used={c}')
                return -1
        elif self.is_transaction_valid(block_obj.transactions[0]) == False:
            parameters.g_logger.debug(f'Block: Invalid First Transaction txn={block_obj.transactions[0]}')
            return -1

        self.cache_update(block_obj.prev_block_hash)

        senders_balance: Dict[str, float] = defaultdict(float)
        senders_balance[block_obj.transactions[0].id_receiver] += block_obj.transactions[0].coin_amount
        if block_obj.transactions[0].id_sender != -1:
            senders_balance[block_obj.transactions[0].id_sender] -= block_obj.transactions[0].coin_amount
        for txn in block_obj.transactions[1:]:
            if txn.id_sender == -1:
                parameters.g_logger.debug(f'Block: Only first transaction can be mining reward transaciton')
                return -1  # Only FIRST transaction can be mining reward transaction
            if txn.id_sender not in self.cache_balance[block_obj.prev_block_hash]:
                parameters.g_logger.debug(f'Cache: Something strange, {txn.id_sender}, {str(self.cache_balance)}, '
                               f'{block_obj.str_all()=}')
            senders_curr_balance: float = senders_balance[txn.id_sender] \
                                          + self.cache_balance[block_obj.prev_block_hash][txn.id_sender]

            # NOTE: we use math.isclose because of limitations of float
            # Example: if sender send 0.13 and calculated current balance is 0.129999999999999
            #          then we must allow it
            if txn.coin_amount > senders_curr_balance and (not math.isclose(txn.coin_amount, senders_curr_balance)):
                parameters.g_logger.debug(f'Block: Invalid transaction txn={txn}')
                # TODO: remove the below "if" statement as it is only for find logical bug in the simulator
                if self.is_transaction_valid(txn, block_obj.prev_block_hash, senders_balance[txn.id_sender]):
                    parameters.g_logger.debug(f'FIXME: cache is not working properly, Cache: Something strange, '
                                   f'{txn.id_sender}, {str(self.cache_balance)}, {block_obj.str_all()=}')
                    continue
                return -1
            # if not self.is_transaction_valid(txn, block_obj.prev_block_hash, senders_balance[txn.id_sender]):
            #     parameters.g_logger.debug(f'Block: Invalid transaction txn={txn}')
            #     return -1
            senders_balance[txn.id_sender] -= txn.coin_amount
            senders_balance[txn.id_receiver] += txn.coin_amount
        return 1

    def block_send(self, block_obj: Block, receiver_node: NodeSiblingInfo):
        self.simulator.event_queue.push(
            Event(
                self.simulator.get_global_time() + receiver_node.find_message_latency(8 * block_obj.size()),
                EventType.EVENT_RECV_BLOCK,
                self.node_id,
                receiver_node.node_id,
                block_obj
            )
        )

    def block_recv(self, block_obj: Block, senders_id: int, current_time: float):
        def m_block_send_all_except(m_block_obj, m_peer_exception_id):
            for m_node in self.neighbors.values():
                # Do NOT send the block back to the node from which it was received
                if m_node.node_id == m_peer_exception_id:
                    continue
                self.block_send(m_block_obj, m_node)

        # IF I have already received the block; THEN I drop it
        if block_obj.curr_block_hash in self.blocks_all:
            return

        # set the block receive time
        block_obj.recv_time = current_time

        # TODO: decide what to do about this, I think the block should not be dropped
        # # IF block_obj.index < max value of current blockchain length; THEN I drop it
        # if block_obj.index < self.blocks_all[self.blockchain_leafs[-1]].index:
        #     return

        block_status: int = self.is_block_valid(block_obj)
        # IF block is invalid; then I drop it
        if block_status == -1:
            parameters.g_logger.warning(f'Problem: Invalid block received')
            return
        # IF parent of the block is not received; then put it in unvalidated blocks list
        if block_status == 0:
            parameters.g_logger.warning(f'Received block cannot be validated because its predecessor is not yet received, '
                             f'node_id={self.node_id}, block={block_obj}')
            self.blocks_unvalidated[block_obj.curr_block_hash] = (senders_id, block_obj,)
            return

        # Store the block and forward it to others
        self.blocks_all[block_obj.curr_block_hash] = block_obj
        m_block_send_all_except(block_obj, senders_id)

        # Check "self.blocks_unvalidated" and updated "block_obj"
        while True:
            end_while_loop = True
            for blk_hash, (blk_sender_id, blk_obj) in self.blocks_unvalidated.items():
                if block_obj.curr_block_hash != blk_obj.prev_block_hash:
                    continue
                # We have already received a successor of the received node
                if self.is_block_valid(blk_obj) == 1:
                    # block is valid
                    end_while_loop = False
                    block_obj = blk_obj
                    self.blocks_all[blk_hash] = blk_obj
                    m_block_send_all_except(blk_obj, blk_sender_id)
                    parameters.g_logger.info(f'node_id={self.node_id}, '
                                  f'Successfully processed an unvalidated block block={blk_obj}')
                    self.blocks_unvalidated.pop(blk_hash)
                else:
                    # block is invalid, pop all blocks which have this block as the ancestor
                    end_while_loop = True
                    successors_1: Set[str] = {blk_hash}
                    successors_2: Set[str] = set()
                    while len(successors_1) != 0:
                        for i in successors_1:
                            successors_2.union(set(filter(
                                lambda x: self.blocks_all[x].prev_block_hash == i,
                                self.blocks_unvalidated
                            )))
                        for i in successors_1:
                            self.blocks_unvalidated.pop(i)
                        successors_1 = successors_2
                        successors_2 = set()
                    pass
                break
            if end_while_loop:
                break

        to_start_new_mining = False
        if block_obj.index > self.blocks_all[self.blockchain_leafs[-1]].index:
            to_start_new_mining = True
            self.last_receive_time = current_time
            parameters.g_logger.info(f'{self.node_id=}, Changing mining branch, '
                          f'new={block_obj.curr_block_hash}, current={self.blockchain_leafs[-1]}')
            self.change_mining_branch(block_obj.curr_block_hash)
            # Insert the block_obj into self.blockchain_leafs
            self.blockchain_leafs.append(block_obj.curr_block_hash)
            self.cache_update(block_obj.curr_block_hash)

        # TODO: see if this is required or not
        # Insert the block_obj into self.blockchain_leafs
        # idx_insert = 0
        # for i in range(len(self.blockchain_leafs) - 1, -1, -1):  # "i" goes from "N-1" to "0"
        #     # NOTE: Mostly, the less than condition will never be true in our simulation
        #     #       However, it can be true in real life
        #     if block_obj.index <= self.blocks_all[self.blockchain_leafs[i]].index:
        #         continue
        #     idx_insert = i + 1
        #     break
        # self.blockchain_leafs.insert(idx_insert, block_obj.curr_block_hash)

        # TODO: This can be removed if all manipulations on "self.blockchain_leafs" are correctly implemented
        # for block in self.blocks_all.values():
        #     try:
        #         self.blockchain_leafs.remove(block.prev_block_hash)
        #         parameters.g_logger.warning(f'')
        #     except ValueError:
        #         pass
        # blockchain_leafs_new = sorted(self.blockchain_leafs, key=lambda x: self.blocks_all[x].index)
        # if self.blockchain_leafs != blockchain_leafs_new:
        #     parameters.g_logger.warning(f'Problem: Blockchain leaf inserting logic not working properly')
        #     parameters.g_logger.warning(f'{self.blockchain_leafs=}')
        #     parameters.g_logger.warning(f'{blockchain_leafs_new}')

        if to_start_new_mining:
            # if block_obj.curr_block_hash not in self.blockchain_leafs:
            #     parameters.g_logger.error('Problem: Latest tail not properly working')
            self.mining_start()
        pass

    def get_new_transaction_greedy(self, curr_tail: str) -> List[Transaction]:
        # NOTE: python indexing [:N] automatically handles the case where length is less than "N"
        # NOTE: we do "self.max_transactions_per_block - 1" because first transaction is Mining Reward Transaction
        self.cache_update(curr_tail)
        txn_list: List[Transaction] = list()
        senders_balance: Dict[int, float] = defaultdict(float)
        for txn in self.txn_pool:
            senders_curr_balance: float = senders_balance[txn.id_sender] + self.cache_balance[curr_tail][txn.id_sender]
            # if self.is_transaction_valid(txn, curr_tail, senders_balance[txn.id_sender]):
            if txn.coin_amount <= senders_curr_balance or math.isclose(txn.coin_amount, senders_curr_balance):
                txn_list.append(txn)
                senders_balance[txn.id_sender] -= txn.coin_amount
                senders_balance[txn.id_receiver] += txn.coin_amount
        return copy.deepcopy(txn_list[:self.max_transactions_per_block - 1])
        # self.txn_pool = list(filter(lambda x: self.is_transaction_valid(x, curr_tail), self.txn_pool))
        # return copy.deepcopy(
        #     self.txn_pool[:self.max_transactions_per_block - 1]
        # )

    def get_new_block(self) -> Tuple[bool, Union[Block, None]]:
        # Point 7 of the PDF
        curr_tail = self.blockchain_leafs[-1]

        sp: SimulatorParameters = self.simulator.simulator_parameters
        txn_mining_reward: Union[Transaction, None] = None
        try:
            txn_mining_reward: Transaction = Transaction(
                self.simulator.get_global_time(),
                -1,
                self.node_id,
                sp.mining_reward_start * (
                        (1 + (sp.mining_reward_update_percent / 100)) **
                        ((self.blocks_all[curr_tail].index + 1) // sp.mining_reward_update_block_time)
                )
            )
        except Exception as e:
            parameters.g_logger.error(e)
            parameters.g_logger.debug(f'curr_tail       = {type(curr_tail)} {curr_tail}')
            parameters.g_logger.debug(
                f'self.blocks_all = {type(self.blocks_all)}, '
                f'{[[str(i), j.str_all()] for i, j in self.blocks_all.items()]}'
            )
            parameters.g_logger.error(traceback.format_exc())
            sys.exit(1)
        transactions_to_include = [txn_mining_reward] + self.get_new_transaction_greedy(curr_tail)

        block_mine_recv_time: float = self.simulator.get_global_time() \
                                      + numpy.random.exponential(self.T_k_exp_block_mining_mean)
        return True, Block(
            curr_tail,
            self.simulator.get_global_time(),
            self.blocks_all[curr_tail].index + 1,
            transactions_to_include,
            block_mine_recv_time,
            block_mine_recv_time
        )

    def mining_start(self) -> bool:
        new_block_status, new_block = self.get_new_block()
        if new_block_status == True and self.is_block_valid(new_block) == False:
            parameters.g_logger.error(f'Problem: FIXME: The block I created is invalid :(')

        if self.is_malicious:
            if new_block_status == False:
                new_block_status = True
                block_mine_recv_time: float = self.simulator.get_global_time() \
                                              + numpy.random.exponential(self.T_k_exp_block_mining_mean)
                new_block = Block(
                    self.blockchain_leafs[-1],
                    self.simulator.get_global_time(),
                    self.blocks_all[self.blockchain_leafs[-1]].index + 1,
                    [Transaction(self.simulator.get_global_time(), -1, self.node_id, 1000)],
                    block_mine_recv_time,
                    block_mine_recv_time
                )
            new_block.transactions[0].coin_amount = 1000.0  # Changing mining reward amount
        if new_block_status == False:
            return False
        self.simulator.event_queue.push(
            Event(
                new_block.recv_time,
                EventType.EVENT_BLOCK_CREATE_SUCCESS,
                self.node_id,
                self.node_id,
                new_block
            )
        )
        return True

    def mining_complete(self, block: Block) -> None:
        # A "Block" which created new longest blockchain was received after the
        # mining started. Hence, we discard this mining complete request because
        # in real life this mining work is to be discarded.
        if block.creation_time < self.last_receive_time:
            # This mining result is to be discarded because we received a new block which makes
            # a longer blockchain after the mining started and before it ended.
            parameters.g_logger.info(
                f'Node Id = {self.node_id}, Mining start time = {block.creation_time:0.5f}, '
                f'Block receive time = {self.last_receive_time:0.5f}, Current time = {self.simulator.get_global_time()}'
            )
            return

        # NOTE: the below two "IF" condition will never be True if `mining_start()` and its depends work correctly
        if self.blocks_all[self.blockchain_leafs[-1]].index > block.index:
            parameters.g_logger.warning(
                f'Problem: node_id={self.node_id}, Block mining complete but a chain '
                f'with longer length is present in "self.blockchain_leafs"'
            )
            parameters.g_logger.warning(f'Problem: node_id={self.node_id}, '
                             f'len queue = {self.blocks_all[self.blockchain_leafs[-1]].index=}, '
                             f'len mined = {self.blocks_all[block.curr_block_hash].index=}')
        if self.blockchain_leafs[-1] != block.prev_block_hash:
            parameters.g_logger.warning(f'Problem: node_id={self.node_id}, '
                             f'len queue = {self.blocks_all[self.blockchain_leafs[-1]].index=}, '
                             f'len mined = {self.blocks_all[block.curr_block_hash].index=}')

        # Add block to the blockchain
        self.blocks_all[block.curr_block_hash] = block
        self.blockchain_leafs[-1] = block.curr_block_hash
        # Remove processed transactions from the transaction pool
        # NOTE: we use [1:] because mining reward transaction will not be in the "self.txn_pool"
        for txn in block.transactions[1:]:
            try:
                self.txn_pool.remove(txn)
            except Exception as e:
                parameters.g_logger.error(e)
                parameters.g_logger.debug(f'txn                = {str(txn)}')
                parameters.g_logger.debug(f'block.transactions = {[str(i) for i in block.transactions]}')
                parameters.g_logger.debug(f'self.txn_pool      = {[str(i) for i in self.txn_pool]}')
                parameters.g_logger.error(traceback.format_exc())
                sys.exit(1)

        # Broadcast the "block" to all "self.neighbors"
        for node in self.neighbors.values():
            self.block_send(block, node)
        self.mining_start()

    def serialize_blockchain_to_str_v1(self) -> str:
        # TODO: update this if required
        return '\n'.join([block.str_all() for block in self.blocks_all.values()])