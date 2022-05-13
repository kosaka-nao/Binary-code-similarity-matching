import abc
import collections
import contextlib
import copy
import random
import time
import six
import networkx as nx
import numpy as np
import tensorflow as tf 

from utils import * 
import pickle 

GraphData = collections.namedtuple('GraphData', [
    'from_idx',
    'to_idx',
    'node_features',
    'edge_features',
    'graph_idx',
    'n_graphs'])


func_struct = collections.namedtuple('func_struct', 'func_name opt_level version compiler bb_size byte_size CFG ARCH LIB_NAME OBF bin_type')

@six.add_metaclass(abc.ABCMeta)
class GraphSimilarityDataset(object):
  """Base class for all the graph similarity learning datasets.

  This class defines some common interfaces a graph similarity dataset can have,
  in particular the functions that creates iterators over pairs and triplets.
  """

  @abc.abstractmethod
  def triplets(self, batch_size):
    """Create an iterator over triplets.

    Args:
      batch_size: int, number of triplets in a batch.

    Yields:
      graphs: a `GraphData` instance.  The batch of triplets put together.  Each
        triplet has 3 graphs (x, y, z).  Here the first graph is duplicated once
        so the graphs for each triplet are ordered as (x, y, x, z) in the batch.
        The batch contains `batch_size` number of triplets, hence `4*batch_size`
        many graphs.
    """
    pass

  @abc.abstractmethod
  def pairs(self, batch_size):
    """Create an iterator over pairs.

    Args:
      batch_size: int, number of pairs in a batch.

    Yields:
      graphs: a `GraphData` instance.  The batch of pairs put together.  Each
        pair has 2 graphs (x, y).  The batch contains `batch_size` number of
        pairs, hence `2*batch_size` many graphs.
      labels: [batch_size] int labels for each pair, +1 for similar, -1 for not.
    """
    pass


class FunctionSimilarityDataset(GraphSimilarityDataset):
  """Graph edit distance dataset."""

  def __init__(self,
               n_nodes_range,
               p_edge_range,
               n_changes_positive,
               n_changes_negative,
               dataset,
               emb_type,
               permute=True
               ):
    """Constructor.

    Args:
      n_nodes_range: a tuple (n_min, n_max).  The minimum and maximum number of
        nodes in a graph to generate.
      p_edge_range: a tuple (p_min, p_max).  The minimum and maximum edge
        probability.
      n_changes_positive: the number of edge substitutions for a pair to be
        considered positive (similar).
      n_changes_negative: the number of edge substitutions for a pair to be
        considered negative (not similar).
      permute: if True (default), permute node orderings in addition to
        changing edges; if False, the node orderings across a pair or triplet of
        graphs will be the same, useful for visualization.
    """
    self._n_min, self._n_max = n_nodes_range
    self._p_min, self._p_max = p_edge_range
    self._k_pos = n_changes_positive
    self._k_neg = n_changes_negative
    self.dataset = dataset 
    self.emb_type = emb_type 
    self._permute = permute
    self.func_set = list(dataset.keys())

  def _set_emb(self,g):
    for id in list(g.nodes.keys()):
        g.node[id]['emb'] = g.node[id][self.emb_type]
        g.node[id]['id'] = id
    return g 
  
  def _get_diff(self,sign,func_struct_ran,cfg_list):
    # if we can get the difference
    plus_diff_delta = 2 
    neg_diff_delta  = 3
    if sign:
      # Get one within delta
      random.shuffle(cfg_list)
      for g in cfg_list:
        diff = abs(func_struct_ran.bb_size - g['cfg'].bb_size)
        if diff >0 and plus_diff_delta>=diff and g['cfg'].bb_size>2:
          g_ret =  self._set_emb(g['cfg'].CFG)
          return g_ret
      # if no, we permute 
      ggg = self._set_emb(func_struct_ran.CFG)
      #permuted_g = permute_graph_nodes(ggg)
      permuted_g = substitute_random_edges(ggg, self._k_pos)
      if permuted_g == False:
          permuted_g = permute_graph_nodes(ggg)
      #print('-'*10)
      return permuted_g
    else:
      ctr =0 
      done = False
      # retry 15 times to get a CFG with atleast neg_diff_delta in bb 
      # Given a large enough dataset, usually it works 
      while done == False and ctr<15:
        ctr += 1
        ran_name = random.choice(self.func_set)
        if ran_name != func_struct_ran.func_name:
          random.shuffle(self.dataset[ran_name])
          for g in self.dataset[ran_name]:
            diff = abs(func_struct_ran.bb_size - g['cfg'].bb_size)
            if diff >0 and diff>=neg_diff_delta and g['cfg'].bb_size>2:
              done = True 
              g_ret =  self._set_emb(g['cfg'].CFG)
              return g_ret


  def _get_graph(self):
    """Generate one graph."""
    # n_nodes = np.random.randint(self._n_min, self._n_max + 1)
    # p_edge = np.random.uniform(self._p_min, self._p_max)

    # # do a little bit of filtering
    # n_trials = 100
    # for _ in range(n_trials):
    #   g = nx.erdos_renyi_graph(n_nodes, p_edge)
    #   if nx.is_connected(g):
    #     for id in list(g.nodes.keys()):
    #         g.node[id]['emb'] = random.random()
    #         g.node[id]['id'] = id 
    #     return g
    # raise ValueError('Failed to generate a connected graph.')
    done = False
    while done == False:
      func = random.choice(self.func_set)
      g = []
      for fucker in self.dataset[func]:
        if fucker['cfg'].CFG.number_of_nodes()>2 and 'diff' in fucker.keys():
          if fucker['diff']['cfg'].CFG.number_of_nodes()>2:
            g.append(fucker)
        elif fucker['cfg'].CFG.number_of_nodes()>2 and 'diff' not in fucker.keys():
          g.append(fucker)
      if len(g)>0:
        done = True
    return g
  
  def do_matching(self):
    assert len(self.dataset) == 2 
    batch_graphs = []
    # We predict both side 
    batch_labels = [1,-1]
    # We skipped the diff mode 
    g_0 = self.dataset[self.func_set[0]][0]
    g_0 = self._set_emb(g_0['cfg'].CFG)
    g_1 = self.dataset[self.func_set[1]][0]
    g_1 = self._set_emb(g_1['cfg'].CFG)
    # Predict twice for 2 sides 
    batch_graphs.append((g_0, g_1))
    batch_graphs.append((g_0, g_1))
    packed_graphs = self._pack_batch(batch_graphs)
    labels = np.array(batch_labels, dtype=np.int32)
    return packed_graphs, labels
  

  def _get_pair(self, positive):
    """Generate one pair of graphs."""
    done = False 
    while done == False:
        g_1 = None 
        g = self._get_graph()
        g_0 = random.choice(g)
        if 'diff' in g_0.keys() and positive:
          g_1 = self._set_emb(g_0['diff']['cfg'].CFG)
        else:
          g_1 = self._get_diff(positive,g_0['cfg'],g)
        g_0 = self._set_emb(g_0['cfg'].CFG)
        if g_0 is not None and g_1 is not None:
            done = True
    # g_1 = self._set_emb(g_1)
    return g_0,g_1
    # done = False 
    # while done == False:
    #     g = self._get_graph()
    #     if len(g)>=2:
    #       done = True
    # if len(g)>=2:
    #   ret = random.sample(g,2)
    #   g_0 = self._set_emb(ret[0]['cfg'].CFG)
    #   g_1 = self._set_emb(ret[1]['cfg'].CFG)
    #   return g_0,g_1
    # else:
    #   g = self._set_emb(g[0]['cfg'].CFG)
    # permuted_g = permute_graph_nodes(g)
    # return g,permuted_g
    '''
    if self._permute:
      permuted_g = permute_graph_nodes(g)
    else:
      permuted_g = g
    n_changes = self._k_pos if positive else self._k_neg
    changed_g = substitute_random_edges(g, n_changes)
    return permuted_g, changed_g
    '''
  def _get_triplet(self):
    """Generate one triplet of graphs."""
    done = False
    while done == False:
        g = self._get_graph()
        g_0 = random.choice(g)
        g_1 = None 
        # we run with diff mode 
        # g_1 = self._get_diff(True ,g_0['cfg'],g)
        if 'diff' in g_0.keys():
          g_1 = self._set_emb(g_0['diff']['cfg'].CFG)
        else:
          g_1 = self._get_diff(True ,g_0['cfg'],g)
        g_2 = self._get_diff(False,g_0['cfg'],g)
        g_0 = self._set_emb(g_0['cfg'].CFG)
        if g_0 is not None and g_1 is not None and g_2 is not None:
            done = True
    # g_1 = self._set_emb(g_1)
    # g_2 = self._set_emb(g_2)
    return g_0,g_1,g_2
    # done = False 
    # while done == False:
    #     g = self._get_graph()
    #     if len(g)>=3:
    #       done = True
    # if len(g)>=3:
    #   ret = random.sample(g,3)
    #   g_0 = self._set_emb(ret[0]['cfg'].CFG)
    #   g_1 = self._set_emb(ret[1]['cfg'].CFG)
    #   g_2 = self._set_emb(ret[2]['cfg'].CFG)
    #   return g_0,g_1,g_2
    # elif len(g)==2:
    #   g_0 = self._set_emb(g[0]['cfg'].CFG)
    #   g_1 = self._set_emb(g[1]['cfg'].CFG)
    #   g_2 = random.choice([g_0,g_1])
    #   g_2 = permute_graph_nodes(g_2)
      #else:
      #  if random.random()>0.5:
      #    g_2 = substitute_random_edges(g_2, self._k_pos)
      #  else:
      #    g_2 = substitute_random_edges(g_2, self._k_neg)
    #   return g_0,g_1,g_2
    # else:
    #   # len(g) == 1
    #   g_0 = self._set_emb(g[0]['cfg'].CFG)
    #   g_1 = permute_graph_nodes(g_0)
    #   g_2 = permute_graph_nodes(g_0)
    #   return g_0,g_1,g_2
      #if self._permute:
      #  permuted_g = permute_graph_nodes(g)
      #else:
      #  permuted_g = g
      #pos_g = substitute_random_edges(g, self._k_pos)
      #neg_g = substitute_random_edges(g, self._k_neg)
      #return permuted_g, pos_g, neg_g

  def triplets(self, batch_size):
    """Yields batches of triplet data."""
    while True:
      batch_graphs = []
      for _ in range(batch_size):
        g1, g2, g3 = self._get_triplet()
        batch_graphs.append((g1, g2, g1, g3))
      yield self._pack_batch(batch_graphs)

  def pairs(self, batch_size):
    """Yields batches of pair data."""
    while True:
      batch_graphs = []
      batch_labels = []
      positive = True
      for _ in range(batch_size):
        g1, g2 = self._get_pair(positive)
        batch_graphs.append((g1, g2))
        batch_labels.append(1 if positive else -1)
        positive = not positive

      packed_graphs = self._pack_batch(batch_graphs)
      labels = np.array(batch_labels, dtype=np.int32)
      yield packed_graphs, labels

  def _pack_batch(self, graphs):
    """Pack a batch of graphs into a single `GraphData` instance.

    Args:
      graphs: a list of generated networkx graphs.

    Returns:
      graph_data: a `GraphData` instance, with node and edge indices properly
        shifted.
    """
    graphs = tf.nest.flatten(graphs)
    from_idx = []
    to_idx = []
    graph_idx = []

    n_total_nodes = 0
    n_total_edges = 0
    emb = []
    for i, g in enumerate(graphs):
      n_nodes = g.number_of_nodes()
      for x in list(g.nodes.keys()): 
        emb.append([g.node[x]['emb']])
      n_edges = g.number_of_edges()
      edges = np.array(g.edges(), dtype=np.int32)
      # shift the node indices for the edges
      from_idx.append(edges[:, 0] + n_total_nodes)
      to_idx.append(edges[:, 1] + n_total_nodes)
      graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)

      n_total_nodes += n_nodes
      n_total_edges += n_edges
      #emb.append(tmp_emb)

    return GraphData(
        from_idx=np.concatenate(from_idx, axis=0),
        to_idx=np.concatenate(to_idx, axis=0),
        # this task only cares about the structures, the graphs have no features
        # TODO 
        #node_features=np.ones((n_total_nodes, 1), dtype=np.float32),
        node_features=np.array(emb),
        edge_features=np.ones((n_total_edges, 1), dtype=np.float32),
        graph_idx=np.concatenate(graph_idx, axis=0),
        n_graphs=len(graphs))


@contextlib.contextmanager
def reset_random_state(seed):
  """This function creates a context that uses the given seed."""
  np_rnd_state = np.random.get_state()
  rnd_state = random.getstate()
  np.random.seed(seed)
  random.seed(seed + 1)
  try:
    yield
  finally:
    random.setstate(rnd_state)
    np.random.set_state(np_rnd_state)


class FixedFunctionSimilarityDataset(FunctionSimilarityDataset):
  """A fixed dataset of pairs or triplets for the graph edit distance task.

  This dataset can be used for evaluation.
  """

  def __init__(self,
               n_nodes_range,
               p_edge_range,
               n_changes_positive,
               n_changes_negative,
               dataset,
               emb_type,
               dataset_size,
               permute=True,
               seed=1234):
    super(FixedFunctionSimilarityDataset, self).__init__(
        n_nodes_range, p_edge_range, n_changes_positive, n_changes_negative,
        dataset=dataset,emb_type=emb_type,permute=permute)
    self._dataset_size = dataset_size
    self._seed = seed

  def triplets(self, batch_size):
    """Yield triplets."""

    if hasattr(self, '_triplets'):
      triplets = self._triplets
    else:
      # get a fixed set of triplets
      with reset_random_state(self._seed):
        triplets = []
        for _ in range(self._dataset_size):
          g1, g2, g3 = self._get_triplet()
          triplets.append((g1, g2, g1, g3))
      self._triplets = triplets

    ptr = 0
    while ptr + batch_size <= len(triplets):
      batch_graphs = triplets[ptr:ptr + batch_size]
      yield self._pack_batch(batch_graphs)
      ptr += batch_size

  def pairs(self, batch_size):
    """Yield pairs and labels."""

    if hasattr(self, '_pairs') and hasattr(self, '_labels'):
      pairs = self._pairs
      labels = self._labels
    else:
      # get a fixed set of pairs first
      with reset_random_state(self._seed):
        pairs = []
        labels = []
        positive = True
        for _ in range(self._dataset_size):
          pairs.append(self._get_pair(positive))
          labels.append(1 if positive else -1)
          positive = not positive
      labels = np.array(labels, dtype=np.int32)

      self._pairs = pairs
      self._labels = labels

    ptr = 0
    while ptr + batch_size <= len(pairs):
      batch_graphs = pairs[ptr:ptr + batch_size]
      packed_batch = self._pack_batch(batch_graphs)
      yield packed_batch, labels[ptr:ptr + batch_size]
      ptr += batch_size
