import random
import time
import simpy
import logging
import copy
import json
import math
import sys
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from transactions import Transaction
from blocks import Block
from network_state_graph import network_creator, csv_loader
from monitor import creater_logger

# turn on/off graphics
graphics = 0

# do the full collision check
full_collision = False

# experiments:
# 0: packet with longest airtime, aloha-style experiment
# 0: one with 3 frequencies, 1 with 1 frequency
# 2: with shortest packets, still aloha-style
# 3: with shortest possible packets depending on distance



# this is an array with measured values for sensitivity
# see paper, Table 3
sf7 = np.array([7,-126.5,-124.25,-120.75])
sf8 = np.array([8,-127.25,-126.75,-124.0])
sf9 = np.array([9,-131.25,-128.25,-127.5])
sf10 = np.array([10,-132.75,-130.25,-128.75])
sf11 = np.array([11,-134.5,-132.75,-128.75])
sf12 = np.array([12,-133.25,-132.25,-132.25])

# Time Frame= 1:10ms
MINING_TIME = 2
BLOCKSIZE = 5
txpool_SIZE = 10
BLOCKTIME = 20
curr = time.ctime()
MESSAGE_COUNT = 0
max_latency = 5
BLOCKID = 99900
#logger.info("-----------------------------------Start of the new Session at %s-------------------------------"%curr)

# check for collisions at base station
# Note: called before a packet (or rather node) is inserted into the list
def checkcollision(packet):
    col = 0 # flag needed since there might be several collisions for packet
    processing = 0
    for i in range(0,len(packetsAtBS)):
        if packetsAtBS[i].packet.processed == 1:
            processing = processing + 1
    if (processing > maxBSReceives):
        #print "too long:", len(packetsAtBS)
        packet.processed = 0
    else:
        packet.processed = 1

    if packetsAtBS:
        #0print "CHECK node {} (sf:{} bw:{} freq:{:.6e}) others: {}".format(packet.nodeid, packet.sf, packet.bw, packet.freq,len(packetsAtBS))
        for other in packetsAtBS:
            if other.nodeid != packet.nodeid:
               #0print ">> node {} (sf:{} bw:{} freq:{:.6e})".format(other.nodeid, other.packet.sf, other.packet.bw, other.packet.freq)
               # simple collision
               if frequencyCollision(packet, other.packet) \
                   and sfCollision(packet, other.packet):
                   if full_collision:
                       if timingCollision(packet, other.packet):
                           # check who collides in the power domain
                           c = powerCollision(packet, other.packet)
                           # mark all the collided packets
                           # either this one, the other one, or both
                           for p in c:
                               p.collided = 1
                               if p == packet:
                                   col = 1
                       else:
                           # no timing collision, all fine
                           pass
                   else:
                       packet.collided = 1
                       other.packet.collided = 1  # other also got lost, if it wasn't lost already
                       col = 1
        return col
    return 0

#
# frequencyCollision, conditions
#
#        |f1-f2| <= 120 kHz if f1 or f2 has bw 500
#        |f1-f2| <= 60 kHz if f1 or f2 has bw 250
#        |f1-f2| <= 30 kHz if f1 or f2 has bw 125
def frequencyCollision(p1,p2):
    if (abs(p1.freq-p2.freq)<=120 and (p1.bw==500 or p2.freq==500)):
        #0print "frequency coll 500"
        return True
    elif (abs(p1.freq-p2.freq)<=60 and (p1.bw==250 or p2.freq==250)):
        #0print "frequency coll 250"
        return True
    else:
        if (abs(p1.freq-p2.freq)<=30):
            #0print "frequency coll 125"
            return True
        #else:
    #0print "no frequency coll"
    return False

def sfCollision(p1, p2):
    if p1.sf == p2.sf:
        #0print "collision sf node {} and node {}".format(p1.nodeid, p2.nodeid)
        # p2 may have been lost too, will be marked by other checks
        return True
    #0print "no sf collision"
    return False

def powerCollision(p1, p2):
    powerThreshold = 6 # dB
    #0print "pwr: node {0.nodeid} {0.rssi:3.2f} dBm node {1.nodeid} {1.rssi:3.2f} dBm; diff {2:3.2f} dBm".format(p1, p2, round(p1.rssi - p2.rssi,2))
    if abs(p1.rssi - p2.rssi) < powerThreshold:
        #0print "collision pwr both node {} and node {}".format(p1.nodeid, p2.nodeid)
        # packets are too close to each other, both collide
        # return both packets as casualties
        return (p1, p2)
    elif p1.rssi - p2.rssi < powerThreshold:
        # p2 overpowered p1, return p1 as casualty
        #0print "collision pwr node {} overpowered node {}".format(p2.nodeid, p1.nodeid)
        return (p1,)
    #0print "p1 wins, p2 lost"
    # p2 was the weaker packet, return it as a casualty
    return (p2,)

def timingCollision(p1, p2):
    # assuming p1 is the freshly arrived packet and this is the last check
    # we've already determined that p1 is a weak packet, so the only
    # way we can win is by being late enough (only the first n - 5 preamble symbols overlap)

    # assuming 8 preamble symbols
    Npream = 8

    # we can lose at most (Npream - 5) * Tsym of our preamble
    Tpreamb = 2**p1.sf/(1.0*p1.bw) * (Npream - 5)

    # check whether p2 ends in p1's critical section
    p2_end = p2.addTime + p2.rectime
    p1_cs = env.now + Tpreamb
    #0print "collision timing node {} ({},{},{}) node {} ({},{})".format(p1.nodeid, env.now - env.now, p1_cs - env.now, p1.rectime,p2.nodeid, p2.addTime - env.now, p2_end - env.now)
    if p1_cs < p2_end:
        # p1 collided with p2 and lost
        #0print "not late enough"
        return True
    #0print "saved by the preamble"
    return False

# this function computes the airtime of a packet
# according to LoraDesignGuide_STD.pdf
#
def airtime(sf,cr,pl,bw):
    H = 0        # implicit header disabled (H=0) or not (H=1)
    DE = 0       # low data rate optimization enabled (=1) or not (=0)
    Npream = 8   # number of preamble symbol (12.25  from Utz paper)

    if bw == 125 and sf in [11, 12]:
        # low data rate optimization mandated for BW125 with SF11 and SF12
        DE = 1
    if sf == 6:
        # can only have implicit header with SF6
        H = 1

    Tsym = (2.0**sf)/bw
    Tpream = (Npream + 4.25)*Tsym
    print "sf", sf, " cr", cr, "pl", pl, "bw", bw
    payloadSymbNB = 8 + max(math.ceil((8.0*pl-4.0*sf+28+16-20*H)/(4.0*(sf-2*DE)))*(cr+4),0)
    Tpayload = payloadSymbNB * Tsym
    return Tpream + Tpayload

#
# this function creates a node
#
class myNode():
    def __init__(self, nodeid, bs, period, packetlen):
        self.nodeid = nodeid
        self.period = period
        self.bs = bs
        self.x = 0
        self.y = 0

        # this is very complex prodecure for placing nodes
        # and ensure minimum distance between each pair of nodes
        found = 0
        rounds = 0
        global loranodes
        while (found == 0 and rounds < 100):
            a = random.random()
            b = random.random()
            if b<a:
                a,b = b,a
            posx = b*maxDist*math.cos(2*math.pi*a/b)+bsx
            posy = b*maxDist*math.sin(2*math.pi*a/b)+bsy
            if len(loranodes) > 0:
                for index, n in enumerate(loranodes):
                    dist = np.sqrt(((abs(n.x-posx))**2)+((abs(n.y-posy))**2))
                    if dist >= 10:
                        found = 1
                        self.x = posx
                        self.y = posy
                    else:
                        rounds = rounds + 1
                        if rounds == 100:
                            print "could not place new node, giving up"
                            exit(-1)
            else:
                print "first node"
                self.x = posx
                self.y = posy
                found = 1
        self.dist = np.sqrt((self.x-bsx)*(self.x-bsx)+(self.y-bsy)*(self.y-bsy))
        print('node %d' %nodeid, "x", self.x, "y", self.y, "dist: ", self.dist)

        self.packet = myPacket(self.nodeid, packetlen, self.dist)
        self.sent = 0

        # graphics for node
        global graphics
        if (graphics == 1):
            global ax
            ax.add_artist(plt.Circle((self.x, self.y), 2, fill=True, color='blue'))

#
# this function creates a packet (associated with a node)
# it also sets all parameters, currently random
#
class myPacket():
    def __init__(self, nodeid, plen, distance):
        global experiment
        global Ptx
        global gamma
        global d0
        global var
        global Lpld0
        global GL

        self.nodeid = nodeid
        self.txpow = Ptx

        # randomize configuration values
        self.sf = random.randint(6,12)
        self.cr = random.randint(1,4)
        self.bw = random.choice([125, 250, 500])

        # for certain experiments override these
        if experiment==1 or experiment == 0:
            self.sf = 12
            self.cr = 4
            self.bw = 125

        # for certain experiments override these
        if experiment==2:
            self.sf = 6
            self.cr = 1
            self.bw = 500
        # lorawan
        if experiment == 4:
            self.sf = 12
            self.cr = 1
            self.bw = 125


        # for experiment 3 find the best setting
        # OBS, some hardcoded values
        Prx = self.txpow  ## zero path loss by default

        # log-shadow
        Lpl = Lpld0 + 10*gamma*math.log10(distance/d0)
        print "Lpl:", Lpl
        Prx = self.txpow - GL - Lpl

        if (experiment == 3) or (experiment == 5):
            minairtime = 9999
            minsf = 0
            minbw = 0

            print "Prx:", Prx

            for i in range(0,6):
                for j in range(1,4):
                    if (sensi[i,j] < Prx):
                        self.sf = int(sensi[i,0])
                        if j==1:
                            self.bw = 125
                        elif j==2:
                            self.bw = 250
                        else:
                            self.bw=500
                        at = airtime(self.sf, 1, plen, self.bw)
                        if at < minairtime:
                            minairtime = at
                            minsf = self.sf
                            minbw = self.bw
                            minsensi = sensi[i, j]
            if (minairtime == 9999):
                print "does not reach base station"
                exit(-1)
            print "best sf:", minsf, " best bw: ", minbw, "best airtime:", minairtime
            self.rectime = minairtime
            self.sf = minsf
            self.bw = minbw
            self.cr = 1

            if experiment == 5:
                # reduce the txpower if there's room left
                self.txpow = max(2, self.txpow - math.floor(Prx - minsensi))
                Prx = self.txpow - GL - Lpl
                print 'minsesi {} best txpow {}'.format(minsensi, self.txpow)

        # transmission range, needs update XXX
        self.transRange = 150
        self.pl = plen
        self.symTime = (2.0**self.sf)/self.bw
        self.arriveTime = 0
        self.rssi = Prx
        # frequencies: lower bound + number of 61 Hz steps
        self.freq = 860000000 + random.randint(0,2622950)

        # for certain experiments override these and
        # choose some random frequences
        if experiment == 1:
            self.freq = random.choice([860000000, 864000000, 868000000])
        else:
            self.freq = 860000000

        print "frequency" ,self.freq, "symTime ", self.symTime
        print "bw", self.bw, "sf", self.sf, "cr", self.cr, "rssi", self.rssi
        self.rectime = airtime(self.sf,self.cr,self.pl,self.bw)
        print "rectime node ", self.nodeid, "  ", self.rectime
        # denote if packet is collided
        self.collided = 0
        self.processed = 0

#
# main discrete event loop, runs for each node
# a global list of packet being processed at the gateway
# is maintained
#
def transmit(env,node):
    while True:
        yield env.timeout(random.expovariate(1.0/float(node.period)))

        # time sending and receiving
        # packet arrives -> add to base station

        node.sent = node.sent + 1
        if (node in packetsAtBS):
            print "ERROR: packet already in"
        else:
            sensitivity = sensi[node.packet.sf - 7, [125,250,500].index(node.packet.bw) + 1]
            if node.packet.rssi < sensitivity:
                print "node {}: packet will be lost".format(node.nodeid)
                node.packet.lost = True
            else:
                node.packet.lost = False
                # adding packet if no collision
                if (checkcollision(node.packet)==1):
                    node.packet.collided = 1
                else:
                    node.packet.collided = 0
                packetsAtBS.append(node)
                node.packet.addTime = env.now

        yield env.timeout(node.packet.rectime)

        if node.packet.lost:
            global nrLost
            nrLost += 1
        if node.packet.collided == 1:
            global nrCollisions
            nrCollisions = nrCollisions +1
        if node.packet.collided == 0 and not node.packet.lost:
            global nrReceived
            nrReceived = nrReceived + 1
        if node.packet.processed == 1:
            global nrProcessed
            nrProcessed = nrProcessed + 1

        # complete packet has been received by base station
        # can remove it
        if (node in packetsAtBS):
            packetsAtBS.remove(node)
            # reset the packet
        node.packet.collided = 0
        node.packet.processed = 0
        node.packet.lost = False


class blocknodes():
    '''
    Properties:
    1. nodeID:      Representing a node

    2. txpool:      A list representing the nodes transaction pool. Its where a new transaction is appended. Acts like a buffer

    3. pendingpool: A list where transaction are stored to form a new block. 
                    Transaction are poped from txpool and are appended to this pool.

    4. block_list:  List of blocks of the node

    5. known_blocks: List of known blocks. It is used for preventing block broadcast forever. It function
                     is defined below in receiver.

    6. known_tx:    List of known Transaction. It is used for preventing block broadcast forever. It function
                     is defined below in receiver.

    7. prev_hash:   Hash of the recent block formed

    8. mine_process: A pointer representing the mining/validator/consensus process. This variable is used to 
                    handle the interrupt.


    '''

    def __init__(self, nodeID):
        self.nodeID = nodeID
        self.env = env
        self.txpool = []
        self.pendingpool = []
        self.block_gas_limit = config['block_gas_limit']
        self.block_list = []
        self.current_gas = 0
        self.current_size = 0
        self.known_blocks = []
        self.known_tx = []
        self.prev_hash = 0
        self.prev_block = 99900
        #self.res= simpy.Resource(env,capacity=1)
        self.miner_flag = 0
        self.broker_status=0
        self.broadcast_domain=0
        self.mine_process = env.process(self.miner())
        #print("Node generated with node ID: %d " % self.nodeID)
        logger.debug('%d,%d, generated, node, -' % (env.now, self.nodeID))

    def add_transaction(self, tx):
        '''
        Method for appending a transaction to the node's transaction pool
        '''
        self.txpool.append(tx)
        self.known_tx.append(tx.id)
        self.broadcaster(tx, self.nodeID, 0, 0)
    '''
     type= 0 :transactions
     type= 1 :blocks
    '''

    def receiver(self, data, type, sent_by):
        '''
        Arguments:
        1. data: The data itself. It could be a block or a transaction.
        2. type: Representation of data. 1 for block, 0 for transaction
        3. sent_by: Sender of the message

        Function of the receiver:
        1. Receive the transactions or blocks broadcasted by other nodes. Check if the transaction was 
           already received by the node; checking if the id of data is present in the known_list. 
            1. If it was previously received, then it was already broadcasted. So no need to broadcast.
            2. Else, broadcast the data to other nodes.

        2. Generate interrupt if a new block is received.
        '''

        global MESSAGE_COUNT
        MESSAGE_COUNT -= 1
        # check if the data is transaction(0) and if the transaction is already included in the blockchain
        if type == 0 and (data.id not in self.known_tx):
            self.txpool.append(data)
            # add the transaction to the known list
            self.known_tx.append(data.id)
            #print("%d received transaction %d at %d"%(self.nodeID,data.id,self.env.now))
            logger.debug("%d,%d,received,transaction,%d " %
                         (self.env.now, self.nodeID, data.id))
            # TODO: comment the below broadcast for all connected network
            # self.broadcaster(data,self.nodeID,0,sent_by)

        # check if the data is block(1) and if the block is already included in the blockchain
        elif type == 1 and (data.id not in self.known_blocks):
            # Use a variable intr_data to store the data for interrupt.
            self.intr_data = data
            # add block to the known list
            self.known_blocks.append(data.id)
            # TODO: comment the below broadcast for all connected network
            # self.broadcaster(data,self.nodeID,1,sent_by)
            #print("%d,%d, received, block, %d"%(self.env.now,self.nodeID,data.id))
            logger.debug("%d,%d, received, block, %d" %
                         (self.env.now, self.nodeID, data.id))
            # Interrupt the mining process
            # self.receive_block()
            self.mine_process.interrupt()
        pass

    def broadcaster(self, data, nodeID, type, sent_by):
        # print("broadcasting")
        # yield env.timeout(1)
        global MESSAGE_COUNT
        global latency
        # Broadcast to neighbour node. For now, broadcast to all.
        #logger.debug('%d , broadcasting, %d'%(self.nodeID,env.now))

        def propagation(delay, each, data, type):
            # TODO: take a gussian time delay
            stat_random = random.gauss(delay, 9)
            if stat_random <= 0:
                stat_random = delay
            yield self.env.timeout(stat_random)
            each.receiver(data, type, nodeID)
        #print("%d, %d, broadcasting, data, %d"%(env.now,self.nodeID,data.id))
        logger.debug("%d, %d, broadcasting, data, %d" %
                     (env.now, self.nodeID, data.id))
        if config['consensus'] != "LMF" :
         for each in node_map:
            # Dont send to self and to the node which sent the message
            if (each.nodeID != self.nodeID) and (each.nodeID != sent_by):
                # insert delay using nodemap
                latency = node_network.loc[self.nodeID, each.nodeID]
                if latency != 0:
                    MESSAGE_COUNT += 1
                    self.env.process(propagation(latency, each, data, type))
                else:
                    pass
          #pass
        else :
         for each in node_map:
            # Dont send to self and to the node which sent the message and sent only to the same broadcast domain
            if (each.nodeID != self.nodeID) and (each.nodeID != sent_by) and (each.broadcast_domain == self.broadcast_domain) :
                # insert delay using nodemap
                latency = node_network.loc[self.nodeID, each.nodeID]
                if latency != 0:
                    MESSAGE_COUNT += 1
                    self.env.process(propagation(latency, each, data, type))
                else:
                    pass
#        return latency
        pass
         

    def miner(self):
        '''
        Block creation method:
        1. For each transaction, add the gas of the transaction to the current gas.
        2. If the current gas is less than block_gas_limit, add more transaction
        3. Else, hold that transaction and create a new block
        4. For new block, store its hash as previous hash, add that block to know list and broadcast 
            it to the other nodes. 

        Interrupt after receiving block from other nodes:
            If a new block is received, the mining process will be interrupted. After interrupt,
            check if the previous block hash of the node matches the previous hash of the block.
        TODO: 
            What to do if prev hash and block id do not match  
        '''
        #print ("first")
        while True:
            try:
                yield env.timeout(0)
                if self.miner_flag == 1:
                    #print ("A0")
                    yield env.timeout(config["mining_time"])
                    if len(self.txpool) != 0:
                        #print ("A1")
                        for each_tx in self.txpool:
                            self.current_gas += each_tx.gas
                            self.current_size += each_tx.size
                            #  Checked: done
                            if self.current_gas < self.block_gas_limit:
                                self.pendingpool.append(self.txpool.pop(0))
                                #print ("A2")   
                            else :
                                #print ("B")
                                break
                    else :
                        #print ("C")
                        pass
                        # could this pass for pending pool be pass by reference ?
                    global BLOCKID
                    BLOCKID += 1
                    self.prev_block += 1
                    block = Block(self.current_size, self.prev_block,
                                  self.pendingpool, self.nodeID, self.prev_hash)
                    self.prev_hash = block.hash
                    print('%d, %d, Created, block, %d,%d' %
                          (env.now, self.nodeID, block.id, block.size))
                    logger.debug('%d, %d, Created, block,%d,%d' %
                                 (env.now, self.nodeID, block.id, block.size))
                    print("hash of block is %s" % block.hash)
                    self.block_list.insert(0, block)
                    block_stability_logger.info("%s,%d,%d,created,%d" % (
                        env.now, self.nodeID, block.id, block.size))
                    network_stability_calc(env, 'c')
                    #print("No of blocks in node %d is %d"%(self.nodeID,len(self.block_list)))
                    logger.info("No of blocks in node %d is %d" %
                                (self.nodeID, len(self.block_list)))
                    self.known_blocks.append(block.id)
                    self.broadcaster(block, self.nodeID, 1, 0)
                    self.current_gas = 0
                    self.current_size = 0
                    self.pendingpool = []
                else:
                    yield env.timeout(0.1)
            except simpy.Interrupt:
                #print("%d,%d, interrupted, block, %d " %(env.now,self.nodeID,self.intr_data.id))
                logger.debug("%d,%d, interrupted, block, %d " %
                             (env.now, self.nodeID, self.intr_data.id))
                # logger.info("testing ...No of blocks in node %d is %d"%(self.nodeID,len(self.block_list)))
                # Verify the block:
                #import ipdb; ipdb.set_trace()
                # check block number
                if self.prev_hash == self.intr_data.prev_hash:
                    #print("Previous hash match")
                    # check the list of transactions
                    block_set = set(self.intr_data.transactions)
                    node_set = set(self.pendingpool)
                    yield env.timeout(config['block_verify_time'])
                    if block_set != node_set:
                        block_extra = block_set-node_set
                        node_extra = node_set-block_set
                        # add item to known tx and transaction pool
                        # Todo : tx id could be repeated in the known tx. Use set for known_tx
                        # print("know tx before extend----> ",self.known_tx)
                        self.known_tx.extend(list(block_extra))
                        # print("know tx after extend----> ",self.known_tx)
                        # move mismatched tx from pendingpool to the txpool
                        self.temp_trans = [
                            each for each in self.pendingpool if each.id in node_extra]
                        self.txpool.extend(self.temp_trans)
                    self.block_list.insert(0, self.intr_data)
                    self.prev_hash = self.intr_data.hash
                    wait = random.randint(0, 45)
                    yield self.env.timeout(wait)
                    block_stability_logger.info("%s,%d,%d,received" % (
                        env.now, self.nodeID, self.intr_data.id))
                    network_stability_calc(env, 'r')
                    #print("No of blocks in node %d is %d"%(self.nodeID,len(self.block_list)))
                    logger.info("No of blocks in node %d is %d" %
                                (self.nodeID, len(self.block_list)))
                    self.pendingpool = []
                    self.intr_data = None
                    self.current_gas = 0
                else:
                    # print("%s,%d,%d,outofsync"%(env.now,self.nodeID,self.intr_data.id))
                    # print(self.prev_hash)
                    # print(self.intr_data.prev_hash)
                    self.prev_hash = self.intr_data.hash
                    block_stability_logger.info("%s,%d,%d,outofsync" % (
                        env.now, self.nodeID, self.intr_data.id))
                    # Simulate node restart by adding the incoming node


def node_generator(env):
    '''
    Generated list of 'n' nodes and network topology with latency using network_creator from 
    network_state_graph based on the parameter NO_NODES
    '''
    global broker_node
    global nodelist
    # load from csv; should create a nodelist and node_map
    if config['load_csv'] == 1:
        global node_network
        node_network, nodelist = csv_loader()
	
    else:
        nodelist = [int(j) for j in range(0,config['n_nodes'])]
        #nodelist = random.sample(range(1000, 1000+config['n_nodes']), config['n_nodes'])
        node_network = network_creator(nodelist, config['max_latency'])
	
    global node_map
    
    node_map = [blocknodes(each) for each in nodelist]
        
    #broker_node=0
    if config['consensus'] == "POW":
        broker_node = random.sample(node_map, config['POW']['miner_number'])
	# Change the sealer_flag for those nodes
        for each in broker_node:
            each.miner_flag = 1
            print("%d selected as miner"%each.nodeID)

    elif config['consensus'] == "POA":
        n_sealer = config['POA']['sealer_number']
        #global broker_node
        broker_node = random.choice(node_map)
        broker_node.miner_flag = 1
        print("Selected sealer is %d" % broker_node.nodeID)
		
    elif config['consensus'] == "LMF":
        n_bd = config['LMF']['bd_number']
        if config['LMF']['bd_number'] > 1:
          broker_node = random.sample(node_map, config['LMF']['bd_number'])
          i=1
          for each in broker_node:
              each.miner_flag = 1
              each.broadcast_domain=i
              #print("For BD %d" % each.broadcast_domain)
              print(" selected broker is %d "%each.nodeID)
              i=i+1
        else :
          broker_node = random.choice(node_map)
          broker_node.miner_flag = 1
          broker_node.broadcast_domain = 1
          print("Selected Broker is %d" % broker_node.nodeID)
        if n_bd <= config['n_nodes'] :
          j=1
          for each in node_map :
              if j <= n_bd :
               if each.broadcast_domain != 0 :
                print("This node is broker, with BD %d" % each.broadcast_domain)
               else :
                each.broadcast_domain=j
                print("Fungsi BD %d" % each.broadcast_domain)
              else :
                j=1
              j=j+1
        else :
            print("Total nodes is %d, Total BD is %d" % config['n_nodes'], n_bd)
            print("Total BD must be smaller than total nodes")
	

def trans_generator(env):
    '''
    1. Generates transaction in a random time derived from Mean Transaction generation time and its 
    Standard Deviation.
    2. Assigns the transaction to a node radomly from the list of transactions.
    '''
    # Use a global ID for transaction
    global txID
    txID = 2300
    if config['LMF']['bd_number'] > 1:
       node = random.choice(broker_node)
    else :
       node = broker_node
    print("Selected sender is %d" % node.nodeID)
    while True:
        # Generate random transaction size and gas
        TX_SIZE = random.gauss(config['mean_tx_size'], config['sd_tx_size'])
        TX_GAS = random.gauss(config['mean_tx_gas'], config['sd_tx_gas'])

        txID += 1
        transaction = Transaction(TX_GAS, TX_SIZE, txID)
        # Choose a node randomly from the nodelist

        # choose a node manually
        #node = 100
        # select a sealer

        # Assign the task to the node; Find the node object with the nodeID
        for i in node_map:

            if i.nodeID == node.nodeID :

                #print("%d, %d, Appended, Transaction, %d"%(env.now,i.nodeID,txID))
                logger.debug("%d, %d,Appended, Transaction, %d" %
                             (env.now, i.nodeID, txID))
                i.add_transaction(transaction)
		
        yield env.timeout(random.gauss(config['mean_tx_generation'], config['sd_tx_generation']))


def monitor(env):
    prev_tx = 2300
    prev_block = 99900
    avg_pending_tx = 0
    while True:
        yield env.timeout(50)
        #print("Current MEssa ges in the system: %d "%MESSAGE_COUNT)
        message_count_logger.info("%d,%d" % (env.now, MESSAGE_COUNT))

        # Transaction per second(Throughput)
        avg_tx = txID-prev_tx
        prev_tx = txID
        # logger.info("%d,%d"%(env.now,avg_tx))

        # tx in pending pool
	#average = 0
        #average = len(node_map.txpool)
        #pending_transaction_logger.info("%d,%d" % (env.now, average))

        # Eventual Consistency # Verified
        hash_list = set()
        len_list = set()
        for each in node_map:
            len_list.add(len(each.block_list))
            for block in each.block_list:
                hash_list.add(block.hash)

        unique_block_logger.info("%d,%d" % (env.now, len(hash_list)))
        #yield average

def network_stability_calc(env, msg):
    #global time_taken
    total = len(nodelist)-1
    if msg == 'c':
        global stb_count, start
        stb_count = 0
        start = env.now
    elif msg == 'r':
        stb_count += 1
        if stb_count >= total:
            time_taken = (env.now - start)*10
            block_creation_logger.info("%s" % (time_taken))
            stb_count = 0
        pass
    #return time_taken

if __name__ == "__main__":
    #env = simpy.rt.RealtimeEnvironment(factor=0.5)
    with open('config.json') as json_data:
        config = json.load(json_data)
    env = simpy.Environment()
    message_count_logger, block_creation_logger, unique_block_logger, pending_transaction_logger, logger, block_stability_logger = creater_logger()
    start_time = time.time()
    print(start_time)
    node_generator(env)
# get arguments
    nrNodes = config['n_nodes']
    avgSendTime = config['mining_time']
    experiment = 5
    simtime = config['sim_time']
    full_collision = bool(1)
    print "Nodes:", nrNodes
    print "AvgSendTime (exp. distributed):",avgSendTime
    print "Experiment: ", experiment
    print "Simtime: ", simtime
    print "Full Collision: ", full_collision
    loranodes = []
    packetsAtBS = []
    maxBSReceives = 8

    # max distance: 300m in city, 3000 m outside (5 km Utz experiment)
    # also more unit-disc like according to Utz
    bsId = 1
    nrCollisions = 0
    nrReceived = 0
    nrProcessed = 0
    nrLost = 0

    Ptx = 14
    gamma = 2.08
    d0 = 40.0
    var = 0           # variance ignored for now
    Lpld0 = 127.41
    GL = 0

    sensi = np.array([sf7,sf8,sf9,sf10,sf11,sf12])
    minsensi = np.amin(sensi) ## Experiment 3 can use any setting, so take minimum
    Lpl = Ptx - minsensi
    print "amin", minsensi, "Lpl", Lpl
    maxDist = d0*(math.e**((Lpl-Lpld0)/(10.0*gamma)))
    print "maxDist:", maxDist
    # base station placement
    bsx = maxDist+10 
    bsy = maxDist+10
    xmax = bsx + maxDist + 20
    ymax = bsy + maxDist + 20

    env.process(trans_generator(env))
    env.process(monitor(env))
    
    for i in range(0,nrNodes):
    # myNode takes period (in ms), base station id packetlen (in Bytes)
    # 1000000 = 16 min
        node = myNode(i,bsId, avgSendTime,20)
        loranodes.append(node)
        env.process(transmit(env,node))

    # env.process(POA(env))
    env.run(until=config['sim_time'])
    # for each in node_map:
    #     #logger.info("Blocks in node %d " %each.nodeID)
    #     print("Blocks in node %d: " %each.nodeID)
    #     for one in each.block_list:
    #         print("Created by %d"%one.generated_by)
    #         one.view_blocks()
    #         #logger.info(one.view_blocks())
    #     print("----------------------------------------------")
    # print stats and save into file
    print "nrCollisions ", nrCollisions
    # compute energy
    # Transmit consumption in mA from -2 to +17 dBm
    TX = [22, 22, 22, 23,                                      # RFO/PA0: -2..1
          24, 24, 24, 25, 25, 25, 25, 26, 31, 32, 34, 35, 44,  # PA_BOOST/PA1: 2..14
          82, 85, 90,                                          # PA_BOOST/PA1: 15..17
          105, 115, 125]                                       # PA_BOOST/PA1+PA2: 18..20
    # mA = 90    # current draw for TX = 17 dBm
    V = 3.0     # voltage XXX
    print node.packet.rectime 
    #print latency
    elapsed_time = time.time() - start_time
 
    sent = sum(n.sent for n in loranodes)
    energy = sum((node.packet.rectime+elapsed_time) * TX[int(node.packet.txpow)+2] * V * node.sent for node in loranodes) / 1e6

    print "energy (in J): ", energy
    print "sent packets: ", sent
    print "collisions: ", nrCollisions
    print "received packets: ", nrReceived
    print "processed packets: ", nrProcessed
    print "lost packets: ", nrLost

    # data extraction rate
    sla = float(100-float((float(sent-nrCollisions)/float(sent))*100))
    print "SLA: ", sla, " % "
    #der = (nrReceived)/float(sent)
    #print "DER method 2:", der

    # save e xperiment data into a dat file that can be read by e.g. gnuplot
    # name of file would be:  exp0.dat for experiment 0
    fname = str(config['consensus']) + "n" + str(experiment) + ".dat"
    print fname
    if os.path.isfile(fname):
       res = "\n" + str(nrNodes) + " " + str(nrCollisions) + " "  + str(sent) + " " + str(energy)+ " " + str(node.packet.rectime+elapsed_time) + " " + str(sla) + " " + str(config['LMF']['bd_number']) + " or " + str(config['POW']['miner_number'])                  
    else:
       res = "#nrNodes nrCollisions nrTransmissions OverallEnergy SentTimes RatioPacketSent NumberMiner/Broker\n" + str(nrNodes) + " " + str(nrCollisions) + " "  + str(sent) + " " + str(energy) + " " + str(node.packet.rectime+elapsed_time) + " " + str(sla) + " " + str(config['LMF']['bd_number']) + " or " + str(config['POW']['miner_number'])        
    with open(fname, "a") as myfile:
         myfile.write(res)
    myfile.close()
    print(elapsed_time)
    print("----------------------------------------------------------------------------------------------")
    print("Simulation ended")
    logger.info("Simulation ended")
    print("Total Time taken %d:" % elapsed_time)
