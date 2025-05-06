import argparse
import torch
import random
import sys
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from config import Config
from copy import deepcopy

from sampler import data_sampler_CFRL
from data_loader import get_data_loader_BERT
from utils import Moment, gen_data
from encoder import EncodingModel
from transformers import BertTokenizer, RobertaTokenizer

import torch.nn.functional as F

import os


class Manager(object):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        
    def _edist(self, x1, x2):
        '''
        input: x1 (B, H), x2 (N, H) ; N is the number of relations
        return: (B, N)
        '''
        b = x1.size()[0]
        L2dist = nn.PairwiseDistance(p=2)
        dist = [] # B
        for i in range(b):
            dist_i = L2dist(x2, x1[i])
            dist.append(torch.unsqueeze(dist_i, 0)) # (N) --> (1,N)
        dist = torch.cat(dist, 0) # (B, N)
        return dist

    def get_memory_proto(self, encoder, dataset):
        '''
        only for one relation data
        '''
        data_loader = get_data_loader_BERT(config, dataset, shuffle=False, \
            drop_last=False,  batch_size=1) 
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance) 
            fea = hidden.detach().cpu().data # (1, H)
            features.append(fea)    
        features = torch.cat(features, dim=0) # (M, H)
        proto = features.mean(0)

        return proto, features   

    def select_memory(self, encoder, dataset):
        '''
        only for one relation data
        '''
        N, M = len(dataset), self.config.memory_size
        data_loader = get_data_loader_BERT(self.config, dataset, shuffle=False, \
            drop_last= False, batch_size=1) # batch_size must = 1
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance) 
            fea = hidden.detach().cpu().data # (1, H)
            features.append(fea)

        features = np.concatenate(features) # tensor-->numpy array; (N, H)
        
        if N <= M: 
            return copy.deepcopy(dataset), torch.from_numpy(features)

        num_clusters = M # memory_size < len(dataset)
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features) # (N, M)

        mem_set = []
        mem_feas = []
        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            sample = dataset[sel_index]
            mem_set.append(sample)
            mem_feas.append(features[sel_index])

        mem_feas = np.stack(mem_feas, axis=0) # (M, H)
        mem_feas = torch.from_numpy(mem_feas)
        # proto = memory mean
        # rel_proto = mem_feas.mean(0)
        # proto = all mean
        features = torch.from_numpy(features) # (N, H) tensor
        rel_proto = features.mean(0) # (H)

        return mem_set, mem_feas
        # return mem_set, features, rel_proto

    def pad_and_truncate(self,sequence, max_length):
        if len(sequence) > max_length:
            return sequence[:max_length]
        else:
            return sequence + [0] * (max_length - len(sequence))

    def data_augmentation(self,memory_samples):
        augmented_data = {}
        keys = memory_samples.keys()
        rel_list =list(keys)
        model_path = self.config.bert_path
        tokenizer = BertTokenizer.from_pretrained(model_path)
        relation_desc_ids= {}
        augment_data = {}
        for rel in rel_list :
            desc = self.r2desc[rel]+' '
            inputs = tokenizer(desc, return_tensors='pt')
            desc_token = tokenizer.encode(' '.join(inputs),
                                        padding='max_length',
                                        truncation=True,
                                        max_length=self.config.max_length)
            desc_token = [x for x in desc_token if x != 0]

            # 提取非0的ids
            for entry in memory_samples[rel]:
                memory_token = [id for id in entry['ids'] if id != 0]
                new_tokens_front = desc_token + memory_token
                new_tokens_back = memory_token + desc_token

                new_tokens_front = self.pad_and_truncate(new_tokens_front, self.config.max_length)
                new_tokens_back = self.pad_and_truncate(new_tokens_back, self.config.max_length)

                # 创建对应的mask
                new_front_mask = [1] * len(new_tokens_front)
                new_front_mask = self.pad_and_truncate(new_front_mask, self.config.max_length)
                new_back_mask = [1] * len(new_tokens_back)
                new_back_mask = self.pad_and_truncate(new_back_mask, self.config.max_length)

                augmented_instance_front = {
                    'relation': entry['relation'],
                    'index': entry['index'],
                    'ids': new_tokens_front,
                    'mask': new_front_mask
                }
                augmented_instance_back = {
                    'relation': entry['relation'],
                    'index': entry['index'],
                    'ids': new_tokens_back,
                    'mask': new_back_mask
                }
                if rel not in augment_data:
                    augment_data[rel] = []
                # augment_data[rel].append(augmented_instance_front)
                augment_data[rel].append(augmented_instance_back)
        return augment_data



    def train_model(self, encoder, training_data, prev_encoder,first_encoder,pre_relationsId,first_relationsId,step, is_memory=False):
        data_loader = get_data_loader_BERT(self.config, training_data, shuffle=True)
        distill_criterion = nn.CosineEmbeddingLoss()
        optimizer = optim.Adam(params=encoder.parameters(), lr=self.config.lr)
        encoder.train()
        weight1 = step
        weight2 = step
        if step > 3 :
           weight1 = weight1+2

        epoch = self.config.epoch_mem if is_memory else self.config.epoch
        for i in range(epoch):
            for batch_num, (instance, labels, ind) in enumerate(data_loader):
                for k in instance.keys():
                    instance[k] = instance[k].to(self.config.device)
                hidden = encoder(instance)
                loss1 = self.moment.contrastive_loss(hidden, labels, is_memory)
                normalized_reps_emb = F.normalize(hidden.view(-1, hidden.size()[1]), p=2, dim=1)
                loss2 = 0.0
                loss3 =0.0
                loss =loss1
                if first_encoder is not None :

                    if labels[0].item() in first_relationsId:
                        first_reps = first_encoder(instance).detach()
                        normalized_first_reps_emb = F.normalize(first_reps.view(-1, first_reps.size()[1]), p=2, dim=1)
                        first_feature_distill_loss = distill_criterion(normalized_reps_emb, normalized_first_reps_emb,
                                                             torch.ones(instance['ids'].size(0)).to(
                                                                 self.config.device))
                        loss2 = first_feature_distill_loss
                    if prev_encoder is not None:
                        if labels[0].item() in pre_relationsId:
                            prev_reps = prev_encoder(instance).detach()
                            normalized_prev_reps_emb = F.normalize(prev_reps.view(-1, prev_reps.size()[1]), p=2, dim=1)

                            feature_distill_loss = distill_criterion(normalized_reps_emb, normalized_prev_reps_emb,
                                                                     torch.ones(instance['ids'].size(0)).to(
                                                                         self.config.device))
                            loss3= feature_distill_loss
                    if step <=4 :
                       loss = loss1 +loss2 * 6 + loss3 *4
                    if step > 4:
                       loss = loss1 +loss2 * 20+ loss3 *6
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # update moment
                if is_memory:
                    self.moment.update(ind, hidden.detach().cpu().data, is_memory=True)
                    # self.moment.update_allmem(encoder)
                else:
                    self.moment.update(ind, hidden.detach().cpu().data, is_memory=False)
                # print
                if is_memory:
                    print('MemoryTrain:  epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                else:
                    print('CurrentTrain: epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
        #         sys.stdout.flush()
        # print('')

    def eval_encoder_proto(self, encoder, seen_proto, seen_relid, test_data,current_relations,pre_relationsId,
                                 first_relationsId):
        batch_size = 1
        test_loader = get_data_loader_BERT(self.config, test_data, False, False, batch_size)
        history_corr =0.0
        first_corrs =0.0
        first_total =0.0
        now_corrs = 0.0
        now_total = 0.0
        corrects = 0.0
        total = 0.0
        encoder.eval()
        current_relationsId =[]
        for rel in current_relations:
            current_relationsId.append(self.rel2id[rel])
        for batch_num, (instance, label, _) in enumerate(test_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance)
            fea = hidden.cpu().data # place in cpu to eval
            logits = -self._edist(fea, seen_proto) # (B, N) ;N is the number of seen relations

            cur_index = torch.argmax(logits, dim=1) # (B)
            pred =  []
            for i in range(cur_index.size()[0]):
                pred.append(seen_relid[int(cur_index[i])])
            pred = torch.tensor(pred)

            correct = torch.eq(pred, label).sum().item()



            acc = correct / batch_size
            if label[0].item() in pre_relationsId:
                now_corrs += correct
                now_total += batch_size
                print('[EVAL] batch: {0:4} | now_acc: {1:3.2f}%,  total now_acc: {2:3.2f}%   ' \
                      .format(batch_num, 100 * acc, 100 * (now_corrs / now_total)) + '\r')
            if label[0].item() in first_relationsId:
                first_corrs += correct
                first_total += batch_size
                print('[EVAL] batch: {0:4} | first_acc: {1:3.2f}%,  total first_acc: {2:3.2f}%   ' \
                      .format(batch_num, 100 * acc, 100 * (first_corrs / first_total)) + '\r')
            corrects += correct
            total += batch_size
            print('[EVAL] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   '\
                .format(batch_num, 100 * acc, 100 * (corrects / total)) + '\r')

        #     sys.stdout.flush()
        # print('')
        if now_total == 0.0:
            now_corrs = corrects
            now_total = total
        return corrects / total,now_corrs/now_total,first_corrs/first_total

    def eval_encoder_proto_1(self, encoder, seen_proto, seen_relid, test_data, current_relations):
        batch_size = 1
        test_loader = get_data_loader_BERT(self.config, test_data, False, False, batch_size)
        history_corr = 0.0
        now_corrs = 0.0
        now_total = 0.0
        corrects = 0.0
        total = 0.0
        encoder.eval()
        current_relationsId = []
        for rel in current_relations:
            current_relationsId.append(self.rel2id[rel])
        for batch_num, (instance, label, _) in enumerate(test_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance)
            fea = hidden.cpu().data  # place in cpu to eval
            logits = -self._edist(fea, seen_proto)  # (B, N) ;N is the number of seen relations

            cur_index = torch.argmax(logits, dim=1)  # (B)
            pred = []
            for i in range(cur_index.size()[0]):
                pred.append(seen_relid[int(cur_index[i])])
            pred = torch.tensor(pred)

            correct = torch.eq(pred, label).sum().item()

            acc = correct / batch_size
            corrects += correct
            total += batch_size
            print('[EVAL] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   ' \
                  .format(batch_num, 100 * acc, 100 * (corrects / total)) + '\r')

        #     sys.stdout.flush()
        # print('')
        return corrects / total

    def _get_sample_text(self, data_path, index):
        sample = {}
        with open(data_path, 'r',encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == index:
                    items = line.strip().split('\t')
                    sample['relation'] = self.id2rel[int(items[0])-1]
                    sample['tokens'] = items[2]
                    sample['h'] = items[3]
                    sample['t'] = items[5]
        return sample

    def _read_description(self, r_path):
        rset = {}
        with open(r_path, 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                rset[items[1]] = items[2]
        return rset


    def train(self):
        # sampler 
        sampler = data_sampler_CFRL(config=self.config, seed=self.config.seed)
        print('prepared data!')
        self.id2rel = sampler.id2rel
        self.rel2id = sampler.rel2id
        self.r2desc = self._read_description(self.config.relation_description)

        # encoder
        encoder = EncodingModel(self.config)

        # step is continual task number
        cur_acc, total_acc,now_acc,first_acc = [], [],[],[]
        cur_acc_num, total_acc_num,now_acc_num,first_acc_num = [], [],[],[]
        memory_samples = {}
        data_generation = []
        history_relations = []
        pre_relationsId = []
        first_relationsId = []
        history_relations_out_first = []
        prev_encoder = None
        first_encoder = None
        for step, (training_data, valid_data, test_data, current_relations, \
            historic_test_data, seen_relations) in enumerate(sampler):
            print(f'current_relations: {current_relations}')
            print(f'seen_relations: {seen_relations}')
            # Initialization
            self.moment = Moment(self.config)

            # Train current task
            training_data_initialize = []

            augment_data = self.data_augmentation(training_data)
            for rel in current_relations:
                training_data_initialize += training_data[rel]
            # if step > 0:
            #     for rel in history_relations:
            #         training_data_initialize += memory_samples[rel]
            self.moment.init_moment(encoder, training_data_initialize, is_memory=False)
            self.train_model(encoder, training_data_initialize,prev_encoder, first_encoder,pre_relationsId,first_relationsId,step)

            if step > 0:
                # memory_data_initialize = []
                # for rel in history_relations:
                #     memory_data_initialize += memory_samples[rel]
                # # memory_data_initialize += data_generation
                # self.moment.init_moment(encoder, memory_data_initialize, is_memory=True)
                # # self.train_model(encoder, memory_data_initialize, None, is_memory=True)
                # self.train_model(encoder, memory_data_initialize, prev_encoder, first_encoder,pre_relationsId,first_relationsId, is_memory=True)
                for rel in current_relations:
                    history_relations_out_first.append(rel)

            # self.train_model(encoder, training_data_initialize, prev_encoder)
            #pre_relations =copy.copy(history_relations)
            # Select memory samples
            for rel in current_relations:
                memory_samples[rel], _ = self.select_memory(encoder, training_data[rel])
                history_relations.append(rel)
            # Data gen
            if self.config.gen == 1:
                gen_text = []
                for rel in current_relations:
                    for sample in memory_samples[rel]:
                        sample_text = self._get_sample_text(self.config.training_data, sample['index'])
                        gen_samples = gen_data(self.r2desc, self.rel2id, sample_text, self.config.num_gen, self.config.gpt_temp, self.config.key)
                        gen_text += gen_samples
                for sample in gen_text:
                    data_generation.append(sampler.tokenize(sample))

            # Data gen
            if step == 0:
                first_encoder = deepcopy(encoder)
                for rel in current_relations:
                    first_relationsId.append(self.rel2id[rel])

                    #Train memory
            if step > 0:
                # augment_data = self.data_augmentation(memory_samples)
                memory_data_initialize = []
                for rel in seen_relations:
                    memory_data_initialize += memory_samples[rel]
                    # memory_data_initialize += augment_data[rel]
                    memory_data_initialize += data_generation
                self.moment.init_moment(encoder, memory_data_initialize, is_memory=True)
                # self.train_model(encoder, memory_data_initialize,None, is_memory=True)
                self.train_model(encoder, memory_data_initialize, prev_encoder, first_encoder, pre_relationsId,
                                 first_relationsId, step,is_memory=True)


            # Update proto
            prev_encoder = deepcopy(encoder)
            seen_proto = []  
            for rel in seen_relations:
                proto, _ = self.get_memory_proto(encoder, memory_samples[rel])
                seen_proto.append(proto)
            seen_proto = torch.stack(seen_proto, dim=0)

            # Eval current task and history task
            test_data_initialize_cur, test_data_initialize_seen = [], []
            for rel in current_relations:
                test_data_initialize_cur += test_data[rel]
            for rel in seen_relations:
                test_data_initialize_seen += historic_test_data[rel]
            seen_relid = []
            for rel in seen_relations:
                seen_relid.append(self.rel2id[rel])
            ac1 = self.eval_encoder_proto_1(encoder, seen_proto, seen_relid, test_data_initialize_cur,current_relations)
            ac2,ac3,ac4= self.eval_encoder_proto(encoder, seen_proto, seen_relid, test_data_initialize_seen,current_relations,pre_relationsId,
                                 first_relationsId)
            cur_acc_num.append(ac1)
            total_acc_num.append(ac2)
            now_acc_num.append(ac3)
            first_acc_num.append(ac4)
            cur_acc.append('{:.4f}'.format(ac1))
            total_acc.append('{:.4f}'.format(ac2))
            now_acc.append('{:.4f}'.format(ac3))
            first_acc.append('{:.4f}'.format(ac4))
            print('cur_acc: ', cur_acc)
            print('now_acc: ', now_acc)
            print('his_acc: ', total_acc)
            print('first_acc: ', first_acc)
            for rel in history_relations_out_first:
                pre_relationsId.append(self.rel2id[rel])

        torch.cuda.empty_cache()
        return total_acc_num,cur_acc_num,now_acc_num,first_acc_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="FewRel", type=str)
    parser.add_argument("--num_k", default=5, type=int)
    parser.add_argument("--num_gen", default=2, type=int)
    args = parser.parse_args()
    config = Config('config.ini')
    config.task_name = args.task_name
    config.num_k = args.num_k
    config.num_gen = args.num_gen

    # config 
    print('#############params############')
    print(config.device)
    config.device = torch.device(config.device)
    print(f'Task={config.task_name}, {config.num_k}-shot')
    print(f'Encoding model: {config.model}')
    print(f'pattern={config.pattern}')
    print(f'mem={config.memory_size}, margin={config.margin}, gen={config.gen}, gen_num={config.num_gen}')
    print('#############params############')

    if config.task_name == 'FewRel':
        config.rel_index = './data/CFRLFewRel/rel_index.npy'
        config.relation_name = './data/CFRLFewRel/relation_name.txt'
        config.relation_description = './data/CFRLFewRel/relation_description.txt'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_5/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/test_0.txt'
    else:
        config.rel_index = './data/CFRLTacred/rel_index.npy'
        config.relation_name = './data/CFRLTacred/relation_name.txt'
        config.relation_description = './data/CFRLTacred/relation_description.txt'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLTacred/CFRLdata_6_100_5_5/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLTacred/CFRLdata_6_100_5_5/train_0.txt'
            config.valid_data = './data/CFRLTacred/CFRLdata_6_100_5_5/valid_0.txt'
            config.test_data = './data/CFRLTacred/CFRLdata_6_100_5_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLTacred/CFRLdata_6_100_5_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLTacred/CFRLdata_6_100_5_10/train_0.txt'
            config.valid_data = './data/CFRLTacred/CFRLdata_6_100_5_10/valid_0.txt'
            config.test_data = './data/CFRLTacred/CFRLdata_6_100_5_10/test_0.txt'        

    # seed 
    random.seed(config.seed) 
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)   
    base_seed = config.seed

    acc_list = []
    cur_acc_list =[]
    now_acc_list = []
    first_acc_list = []
    for i in range(config.total_round):
        config.seed = base_seed + i * 100
        print('--------Round ', i)
        print('seed: ', config.seed)
        manager = Manager(config)
        acc,cur_acc_num,now_acc_num,first_acc_num = manager.train()
        acc_list.append(acc)
        cur_acc_list.append(cur_acc_num)
        now_acc_list.append(now_acc_num)
        first_acc_list.append(first_acc_num)
        torch.cuda.empty_cache()
    
    accs = np.array(acc_list)
    ave = np.mean(accs, axis=0)
    cur_accs = np.array(cur_acc_list)
    cur_ave = np.mean(cur_accs, axis=0)
    now_accs = np.array(now_acc_list)
    now_ave = np.mean(now_accs, axis=0)
    first_accs = np.array(first_acc_list)
    first_ave = np.mean(first_accs, axis=0)
    print('----------END')
    print('cur_acc mean: ', np.around(cur_ave, 4))
    print('pre_acc mean: ', np.around(now_ave, 4))
    print('first_acc mean: ', np.around(first_ave, 4))
    print('his_acc mean: ', np.around(ave, 4))





            
        
            
            


