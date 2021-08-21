#!/usr/bin/env python3
import json
import hashlib
import sys
import math
import random
import coloredlogs
import logging
import numpy
from typing import List, Dict

g_logger = None


class SimulatorParameters:
    # Parameters in the configuration file (will be read from the file during initialization)
    def __init__(self) -> None:
        self.total_nodes = 0
        self.number_of_slow_nodes = 0
        self.number_of_fast_nodes = 0
        self.txn_interarrival_mean = 0
        self.nodes_list: List = []
        self.max_transactions_per_block = 0
        self.node_initial_coins_low = 0  # used to initialize the genesis block
        self.node_initial_coins_high = 0  # used to initialize the genesis block

    # Initialize the simulator parameters
    def initial_setup(self, config_file_name: str):
        global g_logger
        g_logger.debug(f"{config_file_name=}")

        # Open the config file and parse it
        config_file = open(config_file_name)
        parameters = json.load(config_file)

        # Read parameters from the config file

        # Total nodes present in the P2P cryptocurrency network
        self.total_nodes = parameters['total_nodes']

        # parameters["slow-nodes"] === z%
        self.number_of_slow_nodes = (parameters['slow_nodes_percent'] * self.total_nodes) // 100

        if (self.number_of_slow_nodes > self.total_nodes):
            g_logger.error('Condition not satisfied: 0 <= slow_nodes_percent <= 100')
            g_logger.warning('Making all nodes as slow because "slow_nodes_percent > 100"')
            self.number_of_slow_nodes = self.total_nodes

        # (100 - z)% nodes are fast nodes
        self.number_of_fast_nodes = self.total_nodes - self.number_of_slow_nodes

        # Exponential distribution mean for the interarrival between transactions generated by a peer
        self.txn_interarrival_mean = parameters['txn_interarrival_mean']

        self.node_initial_coins_low = parameters['node_initial_coins_low']
        self.node_initial_coins_high = parameters['node_initial_coins_high']
        self.max_transactions_per_block = parameters['max_transactions_per_block']

    # Print all the Simulator Parameters to "stdout"
    def log_parameters(self):
        print(f'Number of peers specified in the config file : {self.total_nodes}')
        print(f'Number of slow nodes : {self.number_of_slow_nodes}')
        print(f'Number of fast nodes : {self.number_of_fast_nodes}')
        print(f'Transaction Interarrival Mean = {self.txn_interarrival_mean}')
        print(f'Nodes List = {self.nodes_list}')


class Simulator:
    def __init__(self, sp: SimulatorParameters):
        self.simulator_parameters = sp
        pass

    def create_genesis_block(self):
        # REFER: https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html#numpy.random.randint
        # “discrete uniform” distribution
        min_block_limit_and_nodes = min(self.simulator_parameters.total_nodes,
                                        self.simulator_parameters.max_transactions_per_block)
        money = numpy.random.randint(
            self.simulator_parameters.node_initial_coins_low,
            self.simulator_parameters.node_initial_coins_high,
            min_block_limit_and_nodes
        )
        node_idx = numpy.random.randint(
            0,
            self.simulator_parameters.total_nodes,
            min_block_limit_and_nodes
        )
        transaction = [Transaction(-1, recv_idx, coins) for recv_idx, coins in zip(node_idx, money)]
        pass


# Structure of a node on the P2P network
class Node:
    def __init__(self, sp: SimulatorParameters):
        # List of connected peers
        self.neighbors = []
        self.txn_interarrival_time = (-1.0 / sp.txn_interarrival_mean) * math.log(random.uniform(0.1, 1))

    # Adding new peer to the neighbors list
    # (inserts an edge between this and new_peer in the node graph)
    def add_new_peer(self, new_peer):
        self.neighbors.append(new_peer)


class Transaction:
    def __init__(self, txn_time, sender, receiver, amount):
        self.txn_time = -1
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.txn_hash = self.get_hash()

    def str(self):
        return str([self.txn_time, self.sender, self.receiver, self.amount])

    def get_hash(self):
        return hashlib.md5(self.str().encode()).hexdigest()


class Block:
    def __init__(self, creating_time=-1, transactions=None):
        if transactions is None:
            transactions = list()
        self.creation_time = creating_time
        self.transactions: List = transactions
        self.block_hash = self.get_hash()

    def update(self, creating_time, transactions: List):
        self.creation_time = creating_time
        self.transactions = transactions
        self.block_hash = self.get_hash()

    def get_hash(self) -> str:
        """
        Hash is calculated based on "self.creation_time" and "self.transactions"
        """
        return hashlib.md5(self.str().encode()).hexdigest()

    def str(self) -> str:
        """
        NOTE: this does not include block hash
        String of "self.creation_time" and "self.transaction"
        """
        return str([self.creation_time, self.transactions])


class EventQueue:
    def __init__(self):
        self.events = list()


def Main(args: Dict):
    ss = SimulatorParameters()
    ss.initial_setup(args['config'])
    ss.log_parameters()
    mySimulator = Simulator(ss)


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
