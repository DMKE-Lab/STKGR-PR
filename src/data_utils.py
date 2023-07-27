"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Data processing utilities.
"""

import collections
import numpy as np
import os
import pickle

START_RELATION = 'START_RELATION'
START_TIME = 'START_TIME'
NO_OP_TIME = 'NO_OP_TIME'
NO_OP_RELATION = 'NO_OP_RELATION'
NO_OP_ENTITY = 'NO_OP_ENTITY'
DUMMY_RELATION = 'DUMMY_RELATION'
DUMMY_ENTITY = 'DUMMY_ENTITY'
DUMMY_TIME = 'DUMMY_TIME'



DUMMY_RELATION_ID = 0
START_RELATION_ID = 1
NO_OP_RELATION_ID = 2
DUMMY_TIME_ID = 0
START_TIME_ID = 1
NO_OP_TIME_ID = 2
DUMMY_ENTITY_ID = 0
NO_OP_ENTITY_ID = 1


def check_answer_ratio(examples):
    entity_dict = {}
    for e1, e2, r in examples:
        if not e1 in entity_dict:
            entity_dict[e1] = set()
        entity_dict[e1].add(e2)
    answer_ratio = 0
    for e1 in entity_dict:
        answer_ratio += len(entity_dict[e1])
    return answer_ratio / len(entity_dict)

def check_relation_answer_ratio(input_file, kg):
    print("@@@")
    example_dict = {}
    with open(input_file) as f:
        for line in f:
            e1, e2, r = line.strip().split()
            e1 = kg.entity2id[e1]
            e2 = kg.entity2id[e2]
            r = kg.relation2id[r]
            if not r in example_dict:
                example_dict[r] = []
            example_dict[r].append((e1, e2, r))
    r_answer_ratio = {}
    for r in example_dict:
        r_answer_ratio[r] = check_answer_ratio(example_dict[r])
    return r_answer_ratio

def change_to_test_model_path(dataset, model_path):
    model_dir = os.path.dirname(os.path.dirname(model_path))
    model_subdir = os.path.basename(os.path.dirname(model_path))
    file_name = os.path.basename(model_path)
    new_model_subdir = dataset + '.test' + model_subdir[len(dataset):]
    new_model_subdir += '-test'
    new_model_path = os.path.join(model_dir, new_model_subdir, file_name)
    return new_model_path

def get_train_path(args):
    if 'NELL' in args.data_dir:
        if not args.model.startswith('point'):
            if args.test:
                train_path = os.path.join(args.data_dir, 'train.dev.large.triples')
            else:
                train_path = os.path.join(args.data_dir, 'train.large.triples')
        else:
            if args.test:
                train_path = os.path.join(args.data_dir, 'train.dev.triples')
            else:
                train_path = os.path.join(args.data_dir, 'train.triples')
    else:
        train_path = os.path.join(args.data_dir, 'train.triples')

    return train_path

def load_seen_entities(adj_list_path, entity_index_path):
    print("((((")
    _, id2entity = load_index(entity_index_path)
    with open(adj_list_path, 'rb') as f:
        adj_list = pickle.load(f)
    seen_entities = set()
    for e1 in adj_list:
        seen_entities.add(id2entity[e1])
        for r in adj_list[e1]:
            for e2 in adj_list[e1][r]:
                seen_entities.add(id2entity[e2])
    print('{} seen entities loaded...'.format(len(seen_entities)))
    return seen_entities
 
def load_triples_with_label(data_path, r, entity_index_path, relation_index_path, seen_entities=None, verbose=False):
    entity2id, _ = load_index(entity_index_path)
    relation2id, _ = load_index(relation_index_path)

    def triple2ids(e1, e2, r):
        return entity2id[e1], entity2id[e2], relation2id[r]

    triples, labels = [], []
    with open(data_path) as f:
        num_skipped = 0
        for line in f:
            pair, label = line.strip().split(': ')
            e1, e2 = pair.strip().split(',')
            if seen_entities and (not e1 in seen_entities or not e2 in seen_entities):
                num_skipped += 1
                if verbose:
                    print('Skip triple ({}) with unseen entity: {}'.format(num_skipped, line.strip())) 
                continue
            triples.append(triple2ids(e1, e2, r))
            labels.append(label.strip())
    return triples, labels

def load_triples(data_path, entity_index_path, relation_index_path, time_index_path, group_examples_by_query=False,
                 add_reverse_relations=False, seen_entities=None, verbose=False, skip_sim=False):
    """
    Convert triples stored on disc into indices.
    """
    entity2id, _ = load_index(entity_index_path)
    relation2id, _ = load_index(relation_index_path)
    time2id, _ = load_index(time_index_path)

    def triple2ids(e1, e2, r, t):
        # print(entity2id[e1], entity2id[e2], relation2id[r], time2id[t])
        return entity2id[e1], entity2id[e2], relation2id[r], time2id[t]

    triples = []
    if group_examples_by_query:
        triple_dict = {}
    with open(data_path) as f:
        num_skipped = 0
        for line in f:
            e1, r, e2, t = line.strip().split("\t")
            if skip_sim and 'SIM' in r:
                continue
            if seen_entities and (not e1 in seen_entities or not e2 in seen_entities):
                num_skipped += 1
                if verbose:
                    print('Skip triple ({}) with unseen entity: {}'.format(num_skipped, line.strip())) 
                continue
            if group_examples_by_query:
                e1_id, e2_id, r_id, t_id = triple2ids(e1, e2, r, t)
                if e1_id not in triple_dict:
                    triple_dict[e1_id] = {}
                if r_id not in triple_dict[e1_id]:
                    triple_dict[e1_id][r_id] = {}
                if t_id not in triple_dict[e1_id][r_id]:
                    triple_dict[e1_id][r_id][t_id] = set()
                triple_dict[e1_id][r_id][t_id].add(e2_id)
                if add_reverse_relations:
                    r_inv = r + '_inv'
                    t_inv = t + '_inv'
                    e2_id, e1_id, r_inv_id, t_inv_id = triple2ids(e2, e1, r_inv, t_inv)
                    if e2_id not in triple_dict:
                        triple_dict[e2_id] = {}
                    if r_inv_id not in triple_dict[e2_id]:
                        triple_dict[e2_id][r_inv_id] = {}
                    if t_inv_id not in triple_dict[e2_id][r_inv_id]:
                        triple_dict[e2_id][r_inv_id][t_inv_id] = set()
                    triple_dict[e2_id][r_inv_id][t_inv_id].add(e1_id)
            else:
                triples.append(triple2ids(e1, e2, r, t))
                if add_reverse_relations:
                    triples.append(triple2ids(e2, e1, r + '_inv', t + '_inv'))
                # if add_reverse_time:
                #     triples.append(triple2ids(e2, e1, r, t + '_inv'))
    if group_examples_by_query:
        for e1_id in triple_dict:
            for r_id in triple_dict[e1_id]:
                for t_id in triple_dict[e1_id][r_id]:
                    triples.append((e1_id, list(triple_dict[e1_id][r_id][t_id]), r_id, t_id))
                    # for x in range(len(triple_dict[e1_id][r_id])):
                    #     triples.append((e1_id, list(triple_dict[e1_id][r_id]), r_id))
    print('{} triples loaded from {}'.format(len(triples), data_path))
    return triples

def load_entity_hist(input_path):
    entity_hist = {}
    with open(input_path) as f:
        for line in f.readlines():
            v, f = line.strip().split()
            entity_hist[v] = int(f)
    return entity_hist

def load_index(input_path):
    index, rev_index = {}, {}
    # print(input_path)
    with open(input_path, encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            # print(line.strip().split("\t"))
            v, _ = line.strip().split("\t")
            # print(v)
            index[v] = i
            rev_index[i] = v
            # print(rev_index, "sss", index)
            # break
    return index, rev_index

def prepare_kb_envrioment(raw_kb_path, train_path, dev_path, test_path, test_mode, add_reverse_relations=True):
    """
    Process KB data which was saved as a set of triples.
        (a) Remove train and test triples from the KB envrionment.
        (b) Add reverse triples on demand.
        (c) Index unique entities and relations appeared in the KB.

    :param raw_kb_path: Path to the raw KB triples.
    :param train_path: Path to the train set KB triples.
    :param dev_path: Path to the dev set KB triples.
    :param test_path: Path to the test set KB triples.
    :param add_reverse_relations: If set, add reverse triples to the KB environment.
    """
    data_dir = os.path.dirname(raw_kb_path)

    def get_type(e_name):
        if e_name == DUMMY_ENTITY:
            return DUMMY_ENTITY
        if 'nell-995' in data_dir.lower():
            if '_' in e_name:
                return e_name.split('_')[1]
            else:
                return 'numerical'
        else:
            return 'entity'

    def hist_to_vocab(_dict):
        return sorted(sorted(_dict.items(), key=lambda x: x[0]), key=lambda x: x[1], reverse=True)

    # Create entity and relation indices
    entity_hist = collections.defaultdict(int)
    relation_hist = collections.defaultdict(int)
    time_hist = collections.defaultdict(int)
    type_hist = collections.defaultdict(int)
    with open(raw_kb_path) as f:
        raw_kb_triples = [l.strip() for l in f.readlines()]
    with open(train_path) as f:
        train_triples = [l.strip() for l in f.readlines()]
    with open(dev_path) as f:
        dev_triples = [l.strip() for l in f.readlines()]
    with open(test_path) as f:
        test_triples = [l.strip() for l in f.readlines()]

    if test_mode:
        keep_triples = train_triples + dev_triples
        removed_triples = test_triples
    else:
        keep_triples = train_triples
        removed_triples = dev_triples + test_triples

    # Index entities and relations and times
    cnt_e = 0
    cnt_r = 0
    cnt_t = 0
    in_l = collections.defaultdict(int)
    out_l = collections.defaultdict(int)
    IN = 0
    OUT = 0
    cnt = 0
    for line in set(raw_kb_triples + keep_triples + removed_triples):
        if len(line.strip().split("\t")) != 4:
            continue
        e1, r, e2, t = line.strip().split("\t")
        if not e1 in out_l:
            OUT += 1
            out_l[e1] = 0
        if not e2 in in_l:
            IN += 1
            in_l[e2] = 0

        if not e1 in entity_hist:
            entity_hist[e1] = cnt_e
            cnt_e += 1
        if not e2 in entity_hist:
            entity_hist[e2] = cnt_e
            cnt_e += 1
        if 'nell-995' in data_dir.lower():
            t1 = e1.split('_')[1] if '_' in e1 else 'numerical'
            t2 = e2.split('_')[1] if '_' in e2 else 'numerical'
        else:
            t1 = get_type(e1)
            t2 = get_type(e2)
        type_hist[t1] += 1
        type_hist[t2] += 1
        if not r in relation_hist:
            relation_hist[r] = cnt_r
        else:
            cnt_r -= 1
        if not t in time_hist:
            time_hist[t] = cnt_t
        else:
            cnt_t -= 1

        if add_reverse_relations:
            inv_r = r + '_inv'
            if not inv_r in relation_hist:
                relation_hist[inv_r] = cnt_r
            inv_t = t + '_inv'
            if not inv_t in time_hist:
                time_hist[inv_t] = cnt_t
        cnt_r += 1
        cnt_t += 1
        cnt += 1
    print("Out-Degree: ", float(cnt) / OUT)
    print("In-Degree: ", float(cnt) / IN)


    # Save the entity and relation indices sorted by decreasing frequency
    with open(os.path.join(data_dir, 'entity2id.txt'), 'w') as o_f:
        o_f.write('{}\t{}\n'.format(DUMMY_ENTITY, DUMMY_ENTITY_ID))
        o_f.write('{}\t{}\n'.format(NO_OP_ENTITY, NO_OP_ENTITY_ID))
        for e, freq in hist_to_vocab(entity_hist):
            o_f.write('{}\t{}\n'.format(e, freq))
    with open(os.path.join(data_dir, 'relation2id.txt'), 'w') as o_f:
        o_f.write('{}\t{}\n'.format(DUMMY_RELATION, DUMMY_RELATION_ID))
        o_f.write('{}\t{}\n'.format(START_RELATION, START_RELATION_ID))
        o_f.write('{}\t{}\n'.format(NO_OP_RELATION, NO_OP_RELATION_ID))
        for r, freq in hist_to_vocab(relation_hist):
            o_f.write('{}\t{}\n'.format(r, freq))

    # 创建time2id
    with open(os.path.join(data_dir, 'time2id.txt'), 'w') as o_f:
        o_f.write('{}\t{}\n'.format(DUMMY_TIME, DUMMY_TIME_ID))
        o_f.write('{}\t{}\n'.format(START_TIME, START_TIME_ID))
        o_f.write('{}\t{}\n'.format(NO_OP_TIME, NO_OP_TIME_ID))
        for r, freq in hist_to_vocab(time_hist):
            o_f.write('{}\t{}\n'.format(r, freq))

    with open(os.path.join(data_dir, 'type2id.txt'), 'w') as o_f:
        for t, freq in hist_to_vocab(type_hist):
            o_f.write('{}\t{}\n'.format(t, freq))
    print('{} entities indexed'.format(len(entity_hist)))
    print('{} relations indexed'.format(len(relation_hist)))
    print('{} times indexed'.format(len(time_hist)))
    print('{} types indexed'.format(len(type_hist)))
    entity2id, id2entity = load_index(os.path.join(data_dir, 'entity2id.txt'))
    relation2id, id2relation = load_index(os.path.join(data_dir, 'relation2id.txt'))
    time2id, id2time = load_index(os.path.join(data_dir, 'time2id.txt'))
    type2id, id2type = load_index(os.path.join(data_dir, 'type2id.txt'))

    removed_triples = set(removed_triples)
    adj_list = collections.defaultdict(lambda: collections.defaultdict(dict))
    entity2typeid = [0 for i in range(len(entity2id))]
    num_facts = 0
    for line in set(raw_kb_triples + keep_triples):
        if len(line.strip().split("\t")) != 4:
            continue
        e1, r, e2, t = line.strip().split("\t")
        triple_signature = '{}\t{}\t{}\t{}'.format(e1, r, e2, t)
        e1_id = entity2id[e1]
        e2_id = entity2id[e2]
        t1 = get_type(e1)
        t2 = get_type(e2)
        t1_id = type2id[t1]
        t2_id = type2id[t2]
        entity2typeid[e1_id] = t1_id
        entity2typeid[e2_id] = t2_id
        if not triple_signature in removed_triples:
            r_id = relation2id[r]
            t_id = time2id[t]
            if not r_id in adj_list[e1_id]:
                adj_list[e1_id][r_id] = dict()

            #下面是自己加的
            if not t_id in adj_list[e1_id][r_id]:
                adj_list[e1_id][r_id][t_id] = set()


            if e2_id in adj_list[e1_id][r_id][t_id]:
                print('Duplicate fact: {} ({}, {}, {}, {})!'.format(
                    line.strip(), id2entity[e1_id], id2relation[r_id], id2time[t_id], id2entity[e2_id]))
            adj_list[e1_id][r_id][t_id].add(e2_id)
            num_facts += 1
            if add_reverse_relations:
                inv_r = r + '_inv'
                inv_t = t + '_inv'
                inv_r_id = relation2id[inv_r]
                inv_t_id = time2id[inv_t]

                if not inv_r_id in adj_list[e2_id]:
                    adj_list[e2_id][inv_r_id] = dict(set([]))

                if not inv_t_id in adj_list[e2_id][inv_r_id]:
                    adj_list[e2_id][inv_r_id][inv_t_id] = set([])

                if e1_id in adj_list[e2_id][inv_r_id][inv_t_id]:
                    print('Duplicate fact: {} ({}, {}, {}, {})!'.format(
                        line.strip(), id2entity[e2_id], id2relation[inv_r_id], id2time[inv_r_id], id2entity[e1_id]))
                adj_list[e2_id][inv_r_id][inv_t_id].add(e1_id)
                num_facts += 1
    print('{} facts processed'.format(num_facts))
    # Save adjacency list
    adj_list_path = os.path.join(data_dir, 'adj_list.pkl')
    with open(adj_list_path, 'wb') as o_f:
        pickle.dump(dict(adj_list), o_f)
    with open(os.path.join(data_dir, 'entity2typeid.pkl'), 'wb') as o_f:
        pickle.dump(entity2typeid, o_f)

def get_seen_queries(data_dir, entity_index_path, relation_index_path, time_index_path):
    entity2id, _ = load_index(entity_index_path)
    relation2id, _ = load_index(relation_index_path)
    time2id, _ = load_index(time_index_path)
    seen_queries = set()
    with open(os.path.join(data_dir, 'train.triples')) as f:
        for line in f:
            e1, r, e2, t = line.strip().split('\t')
            e1_id = entity2id[e1]
            r_id = relation2id[r]
            t_id = time2id[t]
            seen_queries.add((e1_id, r_id, t_id))

    seen_exps = []
    unseen_exps = []
    num_exps = 0
    with open(os.path.join(data_dir, 'dev.triples')) as f:
        for line in f:
            num_exps += 1
            e1, r, e2, t = line.strip().split('\t')
            e1_id = entity2id[e1]
            r_id = relation2id[r]
            t_id = time2id[t]
            if (e1_id, r_id, t_id) in seen_queries:
                seen_exps.append(line)
            else:
                unseen_exps.append(line)
    num_seen_exps = len(seen_exps) + 0.0
    num_unseen_exps = len(unseen_exps) + 0.0
    seen_ratio = num_seen_exps / num_exps
    unseen_ratio = num_unseen_exps / num_exps
    print('Seen examples: {}/{} {}'.format(num_seen_exps, num_exps, seen_ratio))
    print('Unseen examples: {}/{} {}'.format(num_unseen_exps, num_exps, unseen_ratio))

    return seen_queries, (seen_ratio, unseen_ratio)

def get_relations_by_type(data_dir, relation_index_path, time_index_path):
    print("++++")
    with open(os.path.join(data_dir, 'raw.kb')) as f:
        triples = list(f.readlines())
    with open(os.path.join(data_dir, 'train.triples')) as f:
        triples += list(f.readlines())
    # with open(os.path.join(data_dir, 'dev.triples')) as f:
    #     triples += list(f.readlines())
    triples = list(set(triples))

    query_answers = dict()

    theta_1_to_M = 1.5

    for triple_str in triples:
        e1, r, e2, t = triple_str.strip().split('\t')
        if not r in query_answers:
            query_answers[r] = dict()
        if not t in query_answers[r]:
            query_answers[r][t] = dict()
        if not e1 in query_answers[r][t]:
            query_answers[r][t][e1] = set()
        query_answers[r][t][e1].add(e2)

    to_M_rels = set()
    to_1_rels = set()
    to_M_tims = set()
    to_1_tims = set()

    dev_rels = set()
    dev_tims = set()
    with open(os.path.join(data_dir, 'dev.triples')) as f:
        for line in f:
            e1, r, e2, t = line.strip().split('\t')
            dev_rels.add(r)
            dev_tims.add(t)

    relation2id, _ = load_index(relation_index_path)
    time2id, _ = load_index(time_index_path)
    num_rels = len(dev_rels)
    num_tims = len(dev_tims)

    print('{} relations in dev dataset in total'.format(num_rels))
    print('{} tims in dev dataset in total'.format(num_tims))
    for r in dev_rels:
        for t in dev_tims:
            if not r in query_answers or not t in query_answers[r]:
                continue
            ratio = np.mean([len(x) for x in query_answers[r][t].values()])
            if ratio > theta_1_to_M:
                to_M_rels.add(relation2id[r])
                to_M_tims.add(time2id[t])
            else:
                to_1_rels.add(relation2id[r])
                to_1_tims.add(time2id[t])

    num_to_M = len(to_M_rels) + 0.0
    num_to_1 = len(to_1_rels) + 0.0
    num_to_Mt = len(to_M_tims) + 0.0
    num_to_1t = len(to_1_tims) + 0.0

    print('to-M relations: {}/{} ({})'.format(num_to_M, num_rels, num_to_M / num_rels))
    print('to-1 relations: {}/{} ({})'.format(num_to_1, num_rels, num_to_1 / num_rels))
    print('to-M times: {}/{} ({})'.format(num_to_Mt, num_tims, num_to_Mt / num_tims))
    print('to-1 times: {}/{} ({})'.format(num_to_1t, num_tims, num_to_1t / num_tims))

    to_M_examples = []
    to_1_examples = []
    to_Mt_examples = []
    to_1t_examples = []
    num_exps = 0
    with open(os.path.join(data_dir, 'dev.triples')) as f:
        for line in f:
            num_exps += 1
            e1, r, e2, t = line.strip().split('\t')
            if relation2id[r] in to_M_rels:
                to_M_examples.append(line)
            if relation2id[r] in to_1_rels:
                to_1_examples.append(line)
            if time2id[t] in to_M_tims:
                to_Mt_examples.append(line)
            if time2id[t] in to_1_tims:
                to_1t_examples.append(line)
    num_to_M_exps = len(to_M_examples) + 0.0
    num_to_1_exps = len(to_1_examples) + 0.0
    num_to_Mt_exps = len(to_Mt_examples) + 0.0
    num_to_1t_exps = len(to_1t_examples) + 0.0
    to_M_ratio = num_to_M_exps / num_exps
    to_1_ratio = num_to_1_exps / num_exps
    to_Mt_ratio = num_to_Mt_exps / num_exps
    to_1t_ratio = num_to_1t_exps / num_exps
    print('to-M examples: {}/{} ({})'.format(num_to_M_exps, num_exps, to_M_ratio))
    print('to-1 examples: {}/{} ({})'.format(num_to_1_exps, num_exps, to_1_ratio))
    print('to-Mt examples: {}/{} ({})'.format(num_to_Mt_exps, num_exps, to_Mt_ratio))
    print('to-1t examples: {}/{} ({})'.format(num_to_1t_exps, num_exps, to_1t_ratio))
    return to_M_rels, to_1_rels, to_M_tims, to_1_tims, (to_M_ratio, to_1_ratio)

def load_configs(args, config_path):
    with open(config_path) as f:
        print('loading configuration file {}'.format(config_path))
        for line in f:
            if not '=' in line:
                continue
            arg_name, arg_value = line.strip().split('=')
            if arg_value.startswith('"') and arg_value.endswith('"'):
                arg_value = arg_value[1:-1]
            if hasattr(args, arg_name):
                print('{} = {}'.format(arg_name, arg_value))
                arg_value2 = getattr(args, arg_name)
                if type(arg_value2) is str:
                    setattr(args, arg_name, arg_value)
                elif type(arg_value2) is bool:
                    if arg_value == 'True':
                        setattr(args, arg_name, True)
                    elif arg_value == 'False':
                        setattr(args, arg_name, False)
                    else:
                        raise ValueError('Unrecognized boolean value description: {}'.format(arg_value))
                elif type(arg_value2) is int:
                    setattr(args, arg_name, int(arg_value))
                elif type(arg_value2) is float:
                    setattr(args, arg_name, float(arg_value))
                else:
                    raise ValueError('Unrecognized attribute type: {}: {}'.format(arg_name, type(arg_value2)))
            else:
                raise ValueError('Unrecognized argument: {}'.format(arg_name))
    return args
