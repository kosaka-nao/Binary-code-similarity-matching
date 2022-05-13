#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import matplotlib
# matplotlib.use('Agg')
import r2pipe
import json
import pandas as pd
import numpy as np
import glob 
from collections import namedtuple
import networkx as nx
import multiprocessing
import pickle
from gensim.models import Word2Vec,FastText

emb_dict = {'w2v_100_emb' : {'path':'opcode_100.model'},
            'w2v_200_emb' : {'path':'opcode_200.model'},
            'w2v_400_emb' : {'path':'opcode_400.model'},
            'FT_100_emb'  : {'path':'opcode_fasttext_100.model'},
            'FT_200_emb'  : {'path':'opcode_fasttext_200.model'},
            'FT_400_emb'  : {'path':'opcode_fasttext_400.model'}
           }

for key in emb_dict.keys():
    if key.startswith('w2v'):
        model = Word2Vec.load(emb_dict[key]['path'])
    elif key.startswith('FT'):
        model = FastText.load(emb_dict[key]['path'])
    emb_dict[key]['model'] = model 

print("[+] Embedding Loaded ")

def sum_vec(arr):
    '''
    arr with numpy array 
    return the norm 
    '''
    ret = np.copy(arr[0])
    for ix in range(1,len(arr)):
        ret += arr[ix]
    norm = np.linalg.norm(ret)
    return norm

SAVE_PATH = "./"

# TODO 
# Implement as class with customize _eq_ 
func_struct = namedtuple('func_struct', 'func_name opt_level version compiler bb_size byte_size CFG ARCH LIB_NAME OBF bin_type')

def gen_cfg(cfg_json, fname_ = None, for_vec = False, emb_flag = False):
    '''
    cfg_json is the cfg from radare2 
    return networkX graph struct 
    '''
    vect_op = ""
    vect_type = ""
    n = len(cfg_json['blocks'])
    cfg = nx.Graph()
    bb_list = []
    addr_to_ix = {}
    for i in range(0,len(cfg_json['blocks'])):
        tmp_dict = {}
        tmp_dict['opcode'] = []
        tmp_dict['type']   = []
        tmp_dict['disasm'] = []
        fuck_ops = cfg_json['blocks'][i]['ops']
        for j in range(0,len(fuck_ops)):
            if 'opcode' in fuck_ops[j].keys():
                tmp_dict['opcode'].append(fuck_ops[j]['opcode'].split(' ')[0])
            else:
                tmp_dict['opcode'].append("")
            if 'type' in fuck_ops[j].keys():
                tmp_dict['type'].append(fuck_ops[j]['type'])
            else:
                tmp_dict['type'].append("")
            if 'disasm' in fuck_ops[j].keys():
                tmp_dict['disasm'].append(fuck_ops[j]['disasm'])
            else:
                tmp_dict['disasm'].append("")
        if for_vec:
            op_tmp  = " ".join(tmp_dict['opcode'])
            type_tmp = " ".join(tmp_dict['type'])
            vect_op   += op_tmp+"\n"
            vect_type += type_tmp+"\n"
        if emb_flag:
            for key in emb_dict.keys():
                vec_arr = []
                if key.startswith('w2v'):
                    for op in tmp_dict['opcode']:
                        if op in emb_dict[key]['model']:
                            vec_arr.append(emb_dict[key]['model'][op])
                elif key.startswith('FT'):
                    for op in tmp_dict['opcode']:
                        vec_arr.append(emb_dict[key]['model'][op])
                val = 0 
                if len(vec_arr)>0:
                    val = sum_vec(vec_arr)
                tmp_dict[key] = val 
        bb_list.append((i,tmp_dict))
        offset = cfg_json['blocks'][i]['offset']
        addr_to_ix[offset] = i 
    cfg.add_nodes_from(bb_list)
    edge_list = []
    for i in range(0,len(cfg_json['blocks'])):
        offset = cfg_json['blocks'][i]['offset']
        cur_ix = addr_to_ix[offset]
        if 'jump' in cfg_json['blocks'][i].keys():
            jmp_offset = cfg_json['blocks'][i]['jump']
            if jmp_offset in addr_to_ix.keys():
                jmp_ix = addr_to_ix[jmp_offset]
                edge_list.append((cur_ix,jmp_ix))
        if 'fail' in cfg_json['blocks'][i].keys():
            jmp_offset = cfg_json['blocks'][i]['fail']
            if jmp_offset in addr_to_ix.keys():
                jmp_ix = addr_to_ix[jmp_offset]
                edge_list.append((cur_ix,jmp_ix))
    cfg.add_edges_from(edge_list)
    # Plot
    # nx.draw(cfg, with_labels=True)
    # matplotlib.pyplot.savefig('fuck.png', dpi=300, bbox_inches='tight')
    if for_vec:
        fs = open(SAVE_PATH+fname_+"op_vec.log",'a+')
        fs.write(vect_op)
        fs.close()
        fs = open(SAVE_PATH+fname_+"op_type.log",'a+')
        fs.write(vect_type)
        fs.close()
    return cfg 


def analysis(bin_path):
    r2=r2pipe.open(bin_path)
    r2.cmd("aaaaaa")
    #get file type
    # ELF32 or ELF64
    data_json = json.loads(r2.cmd("ij"))
    file_type = data_json["bin"]["class"]
    arch_type = data_json["bin"]["machine"]
    addr_range_json =json.loads(r2.cmd("afllj"))
    addr_size_mapping = {}
    addr_name_mapping = {}
    name_func_struct  = {}
    for i in range(len(addr_range_json)):
        #offset': 134558685, 'name': 'loc.S_0x80532D0', 'size': 41, '
        size = addr_range_json[i]['size']
        if size>=10:
            addr = hex(addr_range_json[i]['offset'])
            name = addr_range_json[i]['name']
            if '.' in name:
                name = name.split('.')[-1]
            addr_size_mapping[str(addr)] = size
            addr_name_mapping[str(addr)] = name
    for x in addr_size_mapping.keys():
        name = addr_name_mapping[str(x)]
        _ = r2.cmd('s '+str(x))
        _ = r2.cmd('s '+str(x))
        done = False 
        retry_ctr = 0 
        while done == False and retry_ctr < 3:
            cfg_json = json.loads(r2.cmd('agfj'))
            if len(cfg_json)>0:
                cfg_json = cfg_json[0]
                done = True 
            else:
                retry_ctr += 1
        if done == True:
            #cfg_json = json.loads(r2.cmd('agfj'))[0]
            fn_ = bin_path.split('/')[-1]
            print('Parsing '+name)
            #try:
            cfg = gen_cfg(cfg_json,fname_ = fn_,for_vec = False,emb_flag = True)
            tmp_func_struct = func_struct(
                func_name = name,
                opt_level = None,
                version = None,
                compiler = None,
                bb_size = len(cfg),
                byte_size = addr_size_mapping[str(x)],
                CFG = cfg, 
                ARCH = arch_type,
                LIB_NAME = None, 
                OBF = False, 
                bin_type = file_type
                )
            name_func_struct[name] = tmp_func_struct
        #except:
        #    print(name+" is failed")
        #    pass 
    r2.quit()
    return name_func_struct


def harness(bin):
    LIB_NAME = 'FFmpeg'
    name_func_struct_tmp = analysis(bin)
    bin_name  = bin.split('/')[-1]
    compiler  = bin_name.split('_')[1]
    opt_level = bin_name.split('_')[2]
    for func_name in name_func_struct_tmp.keys():
        name_func_struct_tmp[func_name]._replace(opt_level=opt_level)
        name_func_struct_tmp[func_name]._replace(compiler=compiler)
        name_func_struct_tmp[func_name]._replace(LIB_NAME=LIB_NAME)
    with open(bin_name+'_cfg.pickle', 'wb') as handle:
        pickle.dump(name_func_struct_tmp, handle, protocol=pickle.HIGHEST_PROTOCOL)

file_list = glob.glob('./ffmpeg_sample/*')
pool = multiprocessing.Pool(int(len(file_list)/2))
ans = pool.map(harness, file_list)

merge = ['ffmpeg_clang_o0_cfg.pickle', 'ffmpeg_clang_o1_cfg.pickle', 'ffmpeg_clang_o2_cfg.pickle', 
         'ffmpeg_clang_o3_cfg.pickle', 'ffmpeg_clang_ofast_cfg.pickle', 'ffmpeg_clang_os_cfg.pickle', 
         'ffmpeg_gcc_o0_cfg.pickle', 'ffmpeg_gcc_o1_cfg.pickle', 'ffmpeg_gcc_o2_cfg.pickle', 
         'ffmpeg_gcc_o3_cfg.pickle', 'ffmpeg_gcc_ofast_cfg.pickle', 'ffmpeg_gcc_os_cfg.pickle', 
         'ffprobe_clang_o0_cfg.pickle', 'ffprobe_clang_o1_cfg.pickle', 'ffprobe_clang_o2_cfg.pickle', 
         'ffprobe_clang_o3_cfg.pickle', 'ffprobe_clang_ofast_cfg.pickle', 'ffprobe_clang_os_cfg.pickle', 
         'ffprobe_gcc_o0_cfg.pickle', 'ffprobe_gcc_o1_cfg.pickle', 'ffprobe_gcc_o2_cfg.pickle',
         'ffprobe_gcc_o3_cfg.pickle', 'ffprobe_gcc_ofast_cfg.pickle', 'ffprobe_gcc_os_cfg.pickle']


def merge_pickle(arr):
    num = ['0','1','2','3','4','5','6','7','8','9']
    query_dict = {}
    dedup_dict = {}
    ret_dict = {}    
    for bin in arr:
        parse_bin = bin.split('/')[-1]
        parse_bin = parse_bin.split('.')[0]
        bin_name  = parse_bin.split('_')[0]
        compiler  = parse_bin.split('_')[1]
        opt_level = parse_bin.split('_')[2]
        name_func_struct_tmp = pickle.load(open(bin, 'rb'))
        for func_name in name_func_struct_tmp.keys():
            # CPP function name can not start with a number
            if func_name[0] not in num:
                random_struct = {'cfg'     : name_func_struct_tmp[func_name],
                                 'compiler': compiler,
                                 'opt'     : opt_level
                                }
                if func_name not in query_dict.keys():
                    query_dict[func_name] = [random_struct]
                else:
                    # Check if sth similar exist before add to the list 
                    add_flag = True 
                    for iter in query_dict[func_name]:
                        if iter['compiler'] == random_struct['compiler'] and \
                            iter['opt'] == random_struct['opt'] and \
                            iter['cfg'].bb_size == random_struct['cfg'].bb_size and\
                            iter['cfg'].byte_size == random_struct['cfg'].byte_size:
                            add_flag = False
                            break 
                    if add_flag:
                        query_dict[func_name].append(random_struct)
    for key in query_dict.keys():
        dedup_dict[key] = []
    for key in query_dict.keys():
        if key.endswith(tuple(num)):
            decom_name = key.split("_")
            decom_name = '_'.join(decom_name[:-1])
            if decom_name in query_dict.keys():
                # sth like this vp56_rac_gets_nn_8
                dedup_dict[decom_name] += query_dict[key]
            else:
                dedup_dict[key] += query_dict[key]
        else:
            dedup_dict[key] += query_dict[key]
    for key in dedup_dict.keys():
        if len(dedup_dict[key])>0:
            arr = []
            fuck_tag_arr = []
            for fuck in dedup_dict[key]:
                fuck_tag = "{}_{}_{}_{}".format(fuck['compiler'],
                                                fuck['opt'],
                                                str(fuck['cfg'].bb_size),
                                                str(fuck['cfg'].byte_size))
                if fuck_tag not in fuck_tag_arr:
                    arr.append(fuck)
                    fuck_tag_arr.append(fuck_tag)
            ret_dict[key] = arr 
    return ret_dict

out_dict = merge_pickle(merge)

with open('ffmpeg_dataset.pickle', 'wb') as handle:
    pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    LIB_NAME = 'FFmpeg'
    file_list = glob.glob('./ffmpeg_sample/*')
    bin_dict = {}
    for bin in file_list:
        name_func_struct_tmp = analysis(bin)
        bin_name  = bin.split('/')[-1]
        compiler  = bin_name.split('_')[1]
        opt_level = bin_name.split('_')[2]
        for func_name in name_func_struct_tmp.keys():
            name_func_struct_tmp[func_name]._replace(opt_level=opt_level)
            name_func_struct_tmp[func_name]._replace(compiler=compiler)
            name_func_struct_tmp[func_name]._replace(LIB_NAME=LIB_NAME)
        bin_dict[bin_name] = name_func_struct_tmp
        with open(bin_name+'_cfg.pickle', 'wb') as handle:
            pickle.dump(name_func_struct_tmp, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('-'*10)

    with open('ffmpeg_db.pickle', 'wb') as handle:
        pickle.dump(bin_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('ffmpeg_db.pickle', 'rb') as handle:
    #     b = pickle.load(handle)

