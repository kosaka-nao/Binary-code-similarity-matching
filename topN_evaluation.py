import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import collections
import copy
import random
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from BinaryFunctionSimilarityDataset import * 
from Graph_model import * 
from dataset import *
import pandas as pd 
import time 

func_struct = collections.namedtuple('func_struct', 'func_name opt_level version compiler bb_size byte_size CFG ARCH LIB_NAME OBF bin_type')

from utils import * 

def build_matchings(layer_outputs, graph_idx, n_graphs, sim):
    """Build the matching attention matrices from layer outputs."""
    assert n_graphs % 2 == 0
    attention = []
    for h in layer_outputs:
        partitions = tf.dynamic_partition(h, graph_idx, n_graphs)
        attention_in_layer = []
        for i in range(0, n_graphs, 2):
            x = partitions[i]
            y = partitions[i + 1]
            a = sim(x, y)
            a_x = tf.nn.softmax(a, axis=1)  # i->j
            a_y = tf.nn.softmax(a, axis=0)  # j->i
            attention_in_layer.append((a_x, a_y))
        attention.append(attention_in_layer)
    return attention


def get_default_config():
  """The default configs."""
  node_state_dim = 32
  graph_rep_dim = 128
  graph_embedding_net_config = dict(
      node_state_dim=node_state_dim,
      edge_hidden_sizes=[node_state_dim * 2, node_state_dim * 2],
      node_hidden_sizes=[node_state_dim * 2],
      n_prop_layers=5,
      # set to False to not share parameters across message passing layers
      share_prop_params=True,
      # initialize message MLP with small parameter weights to prevent
      # aggregated message vectors blowing up, alternatively we could also use
      # e.g. layer normalization to keep the scale of these under control.
      edge_net_init_scale=0.1,
      # other types of update like `mlp` and `residual` can also be used here.
      node_update_type='gru',
      # set to False if your graph already contains edges in both directions.
      use_reverse_direction=True,
      # set to True if your graph is directed
      reverse_dir_param_different=False,
      # we didn't use layer norm in our experiments but sometimes this can help.
      layer_norm=False)
  graph_matching_net_config = graph_embedding_net_config.copy()
  graph_matching_net_config['similarity'] = 'dotproduct'
  return dict(
      encoder=dict(
          node_hidden_sizes=[node_state_dim],
          edge_hidden_sizes=None),
      aggregator=dict(
          node_hidden_sizes=[graph_rep_dim],
          graph_transform_sizes=[graph_rep_dim],
          gated=True,
          aggregation_type='sum'),
      graph_embedding_net=graph_embedding_net_config,
      graph_matching_net=graph_matching_net_config,
      # Set to `embedding` to use the graph embedding net.
      model_type='matching',
      data=dict(
          problem='binary_function_similarity',
          dataset_params=dict(
              # always generate graphs with 20 nodes and p_edge=0.2.
              n_nodes_range=[20, 20],
              p_edge_range=[0.2, 0.2],
              n_changes_positive=1,
              n_changes_negative=2,
              train_set=None,
              validation_set=None,
              emb_type='FT_400_emb',
              validation_dataset_size=1000
              )),
      training=dict(
          batch_size=20,
          learning_rate=1e-3,
          mode='triplet',
          loss='hamming',
          margin=1.0,
          # A small regularizer on the graph vector scales to avoid the graph
          # vectors blowing up.  If numerical issues is particularly bad in the
          # model we can add `snt.LayerNorm` to the outputs of each layer, the
          # aggregated messages and aggregated node representations to
          # keep the network activation scale in a reasonable range.
          graph_vec_regularizer_weight=1e-6,
          # Add gradient clipping to avoid large gradients.
          clip_value=10.0,
          # Increase this to train longer.
          n_training_steps=10000,
          # Print training information every this many training steps.
          print_after=100,
          # Evaluate on validation set every `eval_after * print_after` steps.
          eval_after=10),
      evaluation=dict(
          batch_size=20),
      seed=8,
      )


def fill_dataset(dataset,namelist):
    ret = {}
    for name in namelist:
        ret[name] = dataset[name]
    return ret 

with open('./ffmpeg_dataset_wl_kernel.pickle', 'rb') as handle:
    ffmpeg_dataset = pickle.load(handle)

db_func = list(ffmpeg_dataset.keys())

train_func,test_func = train_test_split(db_func,test_size=0.3, random_state=42)

# train_dict = fill_dataset(ffmpeg_dataset,train_func)
test_dict  = fill_dataset(ffmpeg_dataset,test_func)

std_dataset = {}

for key in test_dict.keys():
    tmp_list_x = test_dict[key]
    random.shuffle(tmp_list_x)
    x = None 
    for z in tmp_list_x:
        if z['cfg'].CFG.number_of_nodes()>2:
            x = z 
            break 
    if x is not None:
        if 'diff' not in x.keys():
            random.shuffle(tmp_list_x)
            against = None 
            for z in tmp_list_x:
                if z['cfg'].CFG.number_of_nodes()>2 and x!=z:
                    against = z 
                    break 
            if against is not None:
                x['diff'] = against
                std_dataset[key] = x 
        else:
            std_dataset[key] = x 



with open('./ffmpeg_topN.pickle', 'wb') as handle:
    pickle.dump(std_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

##################################################################

config = get_default_config()

# Let's just run for a small number of training steps.  This may take you a few
# minutes.
config['training']['n_training_steps'] = 50000
tf.reset_default_graph()

# Set random seeds
seed = config['seed']
random.seed(seed)
np.random.seed(seed + 1)
tf.set_random_seed(seed + 2)

node_feature_dim = 1
edge_feature_dim = 1

tensors, placeholders, model = build_model(
    config, node_feature_dim, edge_feature_dim)

accumulated_metrics = collections.defaultdict(list)

init_ops = (tf.global_variables_initializer(),
            tf.local_variables_initializer())

saver=tf.train.Saver()
sess = tf.Session()
ckpt = tf.train.get_checkpoint_state("./model_ft_400/")    
saver.restore(sess, ckpt.model_checkpoint_path)

ckpt = tf.train.get_checkpoint_state("./model/")    
saver.restore(sess, ckpt.model_checkpoint_path)

with open('./ffmpeg_topN.pickle', 'rb') as handle:
    std_dataset = pickle.load(handle)


def ret_parm(data_dict):
    return dict(
                # always generate graphs with 20 nodes and p_edge=0.2.
                n_nodes_range=[20, 20],
                p_edge_range=[0.2, 0.2],
                n_changes_positive=1,
                n_changes_negative=2,
                dataset=data_dict,
                emb_type='FT_400_emb',
                validation_dataset_size=1000
    )

res_dict = {}

for key_org in std_dataset.keys():
    if std_dataset[key_org]['cfg'].bb_size >1:
        st_time = time.time()
        res_dict[key_org] = {}
        for key_check in std_dataset.keys():
            tmp_dict = {}
            org_object   = copy.copy(std_dataset[key_org])
            if key_org!=key_check:
                check_object = copy.copy(std_dataset[key_check])
                del check_object['diff']
            else:
                check_object = copy.copy(std_dataset[key_check]['diff'])
                key_check += '_self__'
            del org_object['diff']
            tmp_dict[key_org] = [org_object]
            tmp_dict[key_check] = [check_object]
            dataset_params = ret_parm(tmp_dict)
            del dataset_params['validation_dataset_size']
            test_set = FunctionSimilarityDataset(**dataset_params)
            pair_iter = test_set.pairs(1)
            graphs, labels = next(pair_iter)
            n_graphs = graphs.n_graphs
            print(n_graphs)
            model_inputs = placeholders.copy()
            del model_inputs['labels']
            graph_vectors = model(n_graphs=n_graphs, **model_inputs)
            x, y = reshape_and_split_tensor(graph_vectors, 2)
            similarity = compute_similarity(config, x, y)
            layer_outputs = model.get_layer_outputs()
            attentions = build_matchings(
                layer_outputs, placeholders['graph_idx'], n_graphs,
                get_pairwise_similarity(config['graph_matching_net']['similarity']))
            sim, a = sess.run([similarity, attentions],
                            feed_dict=fill_feed_dict(placeholders, (graphs, labels)))
            res_dict[key_org][key_check] = sim[0]
            print("{} vs {} -> {}".format(key_org,key_check,str(sim)))
        d_time = time.time()
        print('It takes '+str(st_time-d_time)+" to match "+key_org)

#----------------------------------------------------------
# single == 2 
# pair == 4 
n_graphs = 4
model_inputs = placeholders.copy()
del model_inputs['labels']
graph_vectors = model(n_graphs=n_graphs, **model_inputs)
x, y = reshape_and_split_tensor(graph_vectors, 2)
similarity = compute_similarity(config, x, y)
layer_outputs = model.get_layer_outputs()
attentions = build_matchings(
    layer_outputs, placeholders['graph_idx'], n_graphs,
    get_pairwise_similarity(config['graph_matching_net']['similarity']))


res_dict = {}

for key_org in std_dataset.keys():
    if std_dataset[key_org]['cfg'].bb_size >1:
        st_time = time.time()
        res_dict[key_org] = {}
        for key_check in std_dataset.keys():
            tmp_dict = {}
            org_object   = copy.copy(std_dataset[key_org])
            if key_org!=key_check:
                check_object = copy.copy(std_dataset[key_check])
                del check_object['diff']
            else:
                check_object = copy.copy(std_dataset[key_check]['diff'])
                key_check += '_self__'
            del org_object['diff']
            tmp_dict[key_org] = [org_object]
            tmp_dict[key_check] = [check_object]
            dataset_params = ret_parm(tmp_dict)
            del dataset_params['validation_dataset_size']
            test_set = FunctionSimilarityDataset(**dataset_params)
            graphs, labels = test_set.do_matching()
            sim, a = sess.run([similarity, attentions],
                            feed_dict=fill_feed_dict(placeholders, (graphs, labels)))
            res_dict[key_org][key_check] = sim[0]
            print("{} vs {} -> {}".format(key_org,key_check,str(sim)))
        d_time = time.time()
        print('It takes '+str(st_time-d_time)+" to match "+key_org)



with open('./scan_res.pickle', 'wb') as handle:
    pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



with open('./ffmpeg_topN.pickle', 'rb') as handle:
    std_dataset = pickle.load(handle)


# Parse 

import pandas as pd  
import pickle 
import copy 


with open('./ffmpeg_topN.pickle', 'rb') as handle:
    std_dataset = pickle.load(handle)

bb_size_dict = {}

for key_org in std_dataset.keys():
    bb_size_dict[key_org] = std_dataset[key_org]['cfg'].bb_size


with open('./scan_res.pickle', 'rb') as handle:
    res_dict = pickle.load(handle)

from_list = []
to_list = []
sim = []
from_bb_size = []
to_bb_size = []


for from_key in res_dict.keys():
    for to_key in res_dict[from_key].keys():
        from_list.append(from_key)
        to_list.append(to_key)
        sim.append(res_dict[from_key][to_key])
        from_bb_size.append(bb_size_dict[from_key])
        to_bb_size.append(bb_size_dict[to_key.replace('_self__','')])


df = pd.DataFrame()

df['from'] = from_list
df['to'] = to_list
df['sim'] = sim
df['from_bb_size'] = from_bb_size
df['to_bb_size'] = to_bb_size



df = df[df['from_bb_size']>=5]
df = df[df['to_bb_size']>=5]

fname = list(set(df['from'].values))

rank = []

for fn in fname:
    try:
        x = df[df['from']==fn]
        x = x.sort_values('sim',ascending=0).reset_index()
        r = x[x['to']==fn+'_self__'].index[0]
        rank.append(r)
    except:
        pass 


from_set = list(set(df['from'].values))

add_list = []

for aaa in from_set:
    add_list.append(aaa+'_self__')

from_set+=add_list

df = df[df['to'].isin(from_set)]

ctr = 0
for p in rank:
    if p  <5:
        ctr+=1

print(ctr)

ctr = 0
for p in rank:
    if p == 0:
        ctr+=1

print(ctr)


from gk_weisfeiler_lehman import GK_WL

graphs = []
graph_name = []
gk_wl = GK_WL()
to_tmp = list(set(df['from'].values))

for sub in to_tmp:
    graphs.append(std_dataset[sub]['cfg'].CFG)
    graph_name.append(sub)

XX = gk_wl.compare_list(graphs, h=1,node_label=False)

with open('./WL_comparsion_sim.pickle', 'wb') as handle:
    pickle.dump(XX, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./WL_comparsion_fnname.pickle', 'wb') as handle:
    pickle.dump(graph_name, handle, protocol=pickle.HIGHEST_PROTOCOL)


##############################################


with open('./WL_comparsion_fnname.pickle', 'rb') as handle:
    graph_name = pickle.load(handle)


fname_to_no = {}

for i in range(0,len(graph_name)):
   fname_to_no[graph_name[i]] = i

fname = list(set(df['from'].values))

rank = []

for fn in fname:
    try:
        x = copy.copy(df[df['from']==fn])
        zzzzzzzz = list(x['to'].values)
        graph_similarity_list = []
        fn_ix = fname_to_no[fn]
        for zzzz in zzzzzzzz:
            if zzzz == fn+'_self__':
                graph_similarity_list.append(0.99)
            else:
                if zzzz in fname_to_no:
                    graph_similarity_list.append(XX[fn_ix][fname_to_no[zzzz]])
                else:
                    graph_similarity_list.append(0)
        x['graph_similarity'] = graph_similarity_list
        x = x[x['graph_similarity']>0.5]
        x = x.sort_values('sim',ascending=0).reset_index()
        r = x[x['to']==fn+'_self__'].index[0]
        rank.append(r)
    except:
        pass 

