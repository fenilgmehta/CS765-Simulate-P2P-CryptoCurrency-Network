#!/usr/bin/env python3
import enum
import json
import hashlib
import sys
import math
import random
import coloredlogs
import logging
import numpy
import copy
import heapq
from typing import List, Dict, Union, Tuple

g_logger = None


class SimulatorParameters:
    # Parameters in the configuration file (will be read from the file during initialization)
    def __init__(self) -> None:
        # Point 1 of PDF: Total Nodes present in the P2P cryptocurrency network
        self.n_total_nodes: int = 0
        # Point 1 of PDF: z% of nodes are slow
        self.z_percent_slow_nodes: float = 0
        # Point 2 of PDF: Exponential Distribution Mean for the inter-arrival
        # time between transactions generated by a peer
        self.T_tx_exp_txn_interarrival_mean_sec: float = 0
        self.T_k_exp_sec_low: float = 0  # Point 7 of PDF: CPU power of the node is high
        self.T_k_exp_sec_high: float = 0  # Point 7 of PDF: CPU power of the node is low
        self.min_light_delay_sec: float = 0.0
        self.max_light_delay_sec: float = 0.0
        self.node_initial_coins_low: float = 0  # used to initialize the genesis block
        self.node_initial_coins_high: float = 0  # used to initialize the genesis block
        self.max_transactions_per_block: int = 0  # Point 7 of PDF: max transactions a block can store

        self.number_of_slow_nodes: int = 0
        self.number_of_fast_nodes: int = 0

    # Initialize the simulator parameters
    def load_from_file(self, config_file_name: str):
        global g_logger
        g_logger.debug(f"{config_file_name=}")

        # Open the config file and parse it
        config_file = open(config_file_name)
        parameters = json.load(config_file)

        # Read parameters from the config file
        self.n_total_nodes: int = int(parameters['n_total_nodes'])
        self.z_percent_slow_nodes: float = float(parameters['z_percent_slow_nodes'])

        # Point 2 of PDF: Exponential Distribution Mean for the inter-arrival
        # time between transactions generated by a peer. Randomly generated
        self.T_tx_exp_txn_interarrival_mean_sec: float = float(parameters['T_tx_exp_txn_interarrival_mean_sec'])
        # Point 7 of PDF: Min and Max time to mine a block by a node
        # This is used to define mining power of each node in the network
        self.T_k_exp_sec_low: float = float(parameters['T_k_exp_sec_low'])
        self.T_k_exp_sec_high: float = float(parameters['T_k_exp_sec_high'])

        self.min_light_delay_sec: float = float(parameters['min_light_delay_sec'])
        self.max_light_delay_sec: float = float(parameters['max_light_delay_sec'])

        self.node_initial_coins_low: float = float(parameters['node_initial_coins_low'])
        self.node_initial_coins_high: float = float(parameters['node_initial_coins_high'])
        self.max_transactions_per_block: int = int(parameters['max_transactions_per_block'])

        # ---
        # z% nodes are slow
        self.number_of_slow_nodes: int = int(self.z_percent_slow_nodes * self.n_total_nodes) // 100
        if self.number_of_slow_nodes > self.n_total_nodes:
            g_logger.error('Condition not satisfied: 0 <= z_percent_slow_nodes <= 100')
            g_logger.warning('Making all nodes as slow because "z_percent_slow_nodes > 100"')
            self.number_of_slow_nodes = self.n_total_nodes

        # (100 - z)% nodes are fast
        self.number_of_fast_nodes: int = self.n_total_nodes - self.number_of_slow_nodes

    # Print all the Simulator Parameters to "stdout"
    def log_parameters(self):
        print()
        print(f'      n  = Number of peers specified in the config file : {self.n_total_nodes}')
        print(f'      z  = {self.z_percent_slow_nodes=} %')
        print(f'      z% = Number of slow nodes : {self.number_of_slow_nodes}')
        print(f'(100-z)% = Number of fast nodes : {self.number_of_fast_nodes}')
        print()
        print(f'    T_tx = (seconds) Exponential Distribution -> '
              f'Transaction Inter-arrival Mean = {self.T_tx_exp_txn_interarrival_mean_sec}')
        print(f' min t_k = (seconds) min block mining time = {self.T_k_exp_sec_low}')
        print(f' max t_k = (seconds) max block mining time = {self.T_k_exp_sec_high}')
        print(f' min ρij = (seconds) min light propagation delay = {self.min_light_delay_sec}')
        print(f' max ρij = (seconds) max light propagation delay = {self.max_light_delay_sec}')
        print()
        print(f'         min initial coins = {self.node_initial_coins_low}')
        print(f'         max initial coins = {self.node_initial_coins_high}')
        print(f'max transactions per block = {self.max_transactions_per_block}')
        print()


class Simulator:
    def __init__(self, sp: SimulatorParameters):
        # TODO
        self.simulator_parameters: SimulatorParameters = sp
        self.global_time: float = 0.0
        self.nodes_list: List[Node] = list()
        self.event_queue: EventQueue = EventQueue()
        pass

    def initialize(self):
        GENESIS_BLOCK = self.__create_genesis_block()
        list_slow_fast: List[bool] = ([False] * self.simulator_parameters.number_of_slow_nodes) \
                                     + ([True] * self.simulator_parameters.number_of_fast_nodes)
        # Create the nodes
        self.nodes_list = [Node(self, self.simulator_parameters, i) for i in list_slow_fast]
        # TODO: Implement point 4 of the problem statement
        #       i.e. Create a connected graph
        pass

    # REFER: https://www.geeksforgeeks.org/private-methods-in-python/
    def __create_genesis_block(self):
        """This method shall only be called during the start of the simulation"""

        # REFER: https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html#numpy.random.randint
        # “discrete uniform” distribution
        min_of_block_limit_and_nodes = min(self.simulator_parameters.n_total_nodes,
                                           self.simulator_parameters.max_transactions_per_block)
        # Uniform random distribution of money
        # In real life, these coins will be with people instead of nodes
        # TODO: change to numpy.random.uniform to generate float value for initial coins a node has
        money = numpy.random.randint(
            self.simulator_parameters.node_initial_coins_low,
            self.simulator_parameters.node_initial_coins_high,
            min_of_block_limit_and_nodes
        )
        # Uniform random distribution to nodes
        node_idx = numpy.random.randint(
            0,
            self.simulator_parameters.n_total_nodes,
            min_of_block_limit_and_nodes
        )
        # Sender = -1 denotes that coins are created from thin air in the genesis block
        transactions = [Transaction(-1, -1, recv_idx, coins) for recv_idx, coins in zip(node_idx, money)]
        return Block('', 0, 0, transactions, 0)

    def get_global_time(self):
        return self.global_time


class Transaction:
    def __init__(self, txn_time: float, id_sender: int, id_receiver: int, coin_amount: float):
        self.txn_time: float = txn_time
        self.id_sender: int = id_sender
        self.id_receiver: int = id_receiver
        self.coin_amount: float = coin_amount
        self.txn_hash: str = self.get_hash()

    def str(self) -> str:
        return str([self.txn_time, self.id_sender, self.id_receiver, self.coin_amount])

    def get_hash(self) -> str:
        return hashlib.md5(self.str().encode()).hexdigest()

    @staticmethod
    def size() -> int:
        """
        Returns size in Bytes
        NOTE: Size is assumed to be 1KB
        """
        return 1000


class Block:
    def __init__(self, prev_block_hash: str, creation_time: float, index: int, transactions: List[Transaction],
                 recv_time: int):
        """self.index is 0-indexed"""
        self.prev_block_hash: str = prev_block_hash
        self.creation_time: float = creation_time
        self.index: int = index
        self.transactions: List[Transaction] = transactions

        # This value/variable is NOT used during hash calculation of this block
        self.curr_block_hash: str = self.get_hash()
        self.recv_time: int = recv_time

    def update(self, prev_block_hash: str, creation_time: float, index: int, transactions: List[Transaction],
               recv_time: int):
        self.prev_block_hash = prev_block_hash
        self.creation_time = creation_time
        self.index = index
        self.transactions = transactions
        self.curr_block_hash = self.get_hash()
        self.recv_time = recv_time

    def get_hash(self) -> str:
        """
        Hash is calculated based on "self.prev_block_hash", "self.creation_time" and "self.transactions"
        """
        return hashlib.md5(self.str().encode()).hexdigest()

    def str(self) -> str:
        """
        NOTE: this does not include block hash
        String of "self.prev_block_hash", "self.creation_time", "self.index" and "self.transaction"
        """
        return str([self.prev_block_hash, self.creation_time, self.index, self.transactions])

    def size(self) -> int:
        # REFER: https://stackoverflow.com/questions/14329794/get-size-in-bytes-needed-for-an-integer-in-python
        # In real life
        # return len(self.prev_block_hash) \
        #        + sys.getsizeof(self.creation_time) \
        #        + sys.getsizeof(self.index) \
        #        + (Transaction.size() * len(self.transactions))
        # In our simulator
        return Transaction.size() * len(self.transactions)


class Node:
    """Structure of a node on the P2P network"""

    def __init__(self, simulator: Simulator, sp: SimulatorParameters, is_network_fast: bool):
        self.simulator: Simulator = simulator
        self.is_network_fast = is_network_fast

        # List of connected peers
        self.neighbors: List[Node.NodeSiblingInfo] = []
        # The time at which a new block with chain lenght > local chain length is received
        self.last_receive_time: int = -1

        self.txn_all: Dict[str, Transaction] = dict()
        self.txn_pool: List[Transaction] = list()

        self.blocks_all: Dict[str, Block] = dict()
        self.block_chain_leafs: List[str] = list()

        # Point 7 of PDF: Exponential Distribution Mean for the
        # block mining time by node. It is randomly generated
        # from Simulator Parameters "sp.T_k_exp_sec_low" and "sp.T_k_exp_sec_high"
        # Lower value  => High CPU power
        # Higher value => Low CPU power
        self.T_k_exp_block_mining_mean = numpy.random.uniform(sp.T_k_exp_sec_low, sp.T_k_exp_sec_high)

        # This is same for all nodes
        # Point 2 of PDF: Exponential Distribution Mean for inter-arrival time between transaction
        self.T_tx_exp_txn_interarrival_mean_sec = sp.T_tx_exp_txn_interarrival_mean_sec

        # This is same for all nodes
        # Max transactions a block can store
        self.max_transactions_per_block = sp.max_transactions_per_block

    def add_new_peer(self, new_peer_id: int, node_id: int, rho_ij: float, c_ij: int) -> None:
        """
        Adding new peer to the neighbors list
        Inserts an edge between this and new_peer in the node graph
        """
        self.neighbors.append(Node.NodeSiblingInfo(new_peer_id, rho_ij, c_ij))

    def remove_peer(self, peer_id: int) -> bool:
        for i in range(len(self.neighbors)):
            if self.neighbors[i].node_id == peer_id:
                self.neighbors.pop(i)
                return True
        return False

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
            d_ij = numpy.random.exponential(96_000 / self.c_ij)  # TODO
            return self.rho_ij + (message_size_bits / self.c_ij) + d_ij

    def transaction_validate(self, transaction_obj):
        # TODO
        pass

    def transaction_send(self, transaction_obj: Transaction, receivers_id: int):
        self.simulator.event_queue.push(
            Event(self.simulator.get_global_time(), EventType.EVENT_RECV_TRANSACTION, transaction_obj)
        )

    def transaction_recv(self, transaction_obj: Transaction, senders_id: int):
        # IF I have already received the transaction; THEN I drop it
        if transaction_obj.txn_hash in self.txn_all:
            return
        # IF transaction is invalid; then I drop it
        if self.transaction_validate(transaction_obj) == False:
            return
        self.txn_all[transaction_obj.txn_hash] = transaction_obj
        for node in self.neighbors:
            # Do NOT send the transaction back to the node from which it was received
            if node.node_id == senders_id:
                continue
            self.transaction_send(transaction_obj, node.node_id)
        pass

    def block_validate(self, block_obj):
        # TODO
        pass

    def block_send(self, block_obj: Block, receivers_id: int):
        # TODO
        pass

    def block_recv(self, block_obj: Block, senders_id: int, current_time: float):
        block_obj.recv_time = current_time
        # IF I have already received the block; THEN I drop it
        if block_obj.curr_block_hash in self.blocks_all:
            return
        # IF block is invalid; then I drop it
        if self.block_validate(block_obj) == False:
            return
        self.blocks_all[block_obj.curr_block_hash] = block_obj
        for node in self.neighbors:
            if node.node_id == senders_id:
                continue
            self.block_send(block_obj, node.node_id)
        # TODO - other condition checks
        pass

    def broadcast(self, obj_to_broadcast):
        # TODO
        pass


# REFER: https://www.tutorialspoint.com/enum-in-python
class EventType(enum.Enum):
    EVENT_UNDEFINED = 0
    EVENT_SEND_TRANSACTION = 1
    EVENT_RECV_TRANSACTION = 2
    EVENT_SEND_BLOCK = 3
    EVENT_RECV_BLOCK = 4


class Event:
    def __init__(self, event_time: float, event_type: EventType, data_obj):
        self.event_time: float = 0.0
        self.event_type: EventType = EventType.EVENT_UNDEFINED
        self.data_obj = data_obj


# REFER: https://docs.python.org/3/library/heapq.html
# REFER: https://www.geeksforgeeks.org/heap-queue-or-heapq-in-python/
class EventQueue:
    def __init__(self):
        self.events: List[Tuple[float, Event]] = list()

    def push(self, new_event: Event):
        heapq.heappush(self.events, (new_event.event_time, new_event))

    def pop(self) -> Event:
        return heapq.heappop(self.events)[1]


def Main(args: Dict):
    ss = SimulatorParameters()
    ss.load_from_file(args['config'])
    ss.log_parameters()
    mySimulator = Simulator(ss)
    mySimulator.initialize()
    pass


if __name__ == '__main__':
    import argparse

    my_parser = argparse.ArgumentParser(prog='simulator.py',
                                        description='Discrete Event Simulator for a P2P Cryptocurrency Network',
                                        epilog='Enjoy the program :)',
                                        prefix_chars='-',
                                        allow_abbrev=False,
                                        add_help=True)
    my_parser.version = '1.0'
    my_parser.add_argument('--version', action='version')
    my_parser.add_argument('-D',
                           '--debug',
                           action='store_true',
                           help='Print debug information')
    my_parser.add_argument('-c',
                           '--config',
                           action='store',
                           type=str,
                           help='Path to the config file',
                           required=True)

    # Execute the parse_args() method
    args: argparse.Namespace = my_parser.parse_args()

    # Initialize "g_logger"
    g_logger = logging.getLogger(__name__)
    # REFER: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    #            - https://github.com/xolox/python-coloredlogs
    if args.debug:
        # g_logger.setLevel(logging.DEBUG)
        coloredlogs.install(fmt='%(levelname)-8s :: [%(lineno)4s] %(name)10s :: %(message)s', level='DEBUG',
                            logger=g_logger)
    else:
        # g_logger.setLevel(logging.INFO)
        coloredlogs.install(fmt='%(levelname)-8s :: [%(lineno)4s] %(name)10s :: %(message)s', level='INFO',
                            logger=g_logger)

    g_logger.debug('Debugging is ON')
    g_logger.debug(f'{args=}')
    g_logger.handlers[
        0].flush()  # REFER: https://stackoverflow.com/questions/13176173/python-how-to-flush-the-log-django/13753911

    Main(vars(args))
