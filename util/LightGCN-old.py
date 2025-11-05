import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
# from util.loss_torch import bpr_loss,l2_reg_loss, bpr_k_loss, ccl_loss, directau_loss, gbce_loss, simce_loss, ssm_loss
from util.loss_torch import bpr_loss,l2_reg_loss, bpr_k_loss, ccl_loss, directau_loss, simce_loss, ssm_loss
import torch.nn.functional as F
from fkan.torch import FractionalJacobiNeuralBlock as fJNB
from sklearn.cluster import KMeans

from util.args import get_params
import time


args = get_params()

class LightGCN(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(LightGCN, self).__init__(conf, training_set, test_set)
        _args = OptionConf(self.config['LightGCN'])
        self.n_layers = int(_args['-n_layer'])
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)
        if args.loss in ['ssm', 'simce']:
            self.maxEpoch = 50

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            t0 = time.time()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):

                user_idx, pos_idx, neg_idx = batch

                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                        neg_idx]

                reg_loss = l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb) / self.batch_size

                if args.loss == 'bpr':
                    rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                elif args.loss == 'ssm':
                    rec_loss = ssm_loss(user_emb, pos_item_emb, neg_item_emb)
                elif args.loss == 'simce':
                    rec_loss = simce_loss(user_emb, pos_item_emb, neg_item_emb, margin=args.margin)

                batch_loss = rec_loss + reg_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                # if n % 100==0 and n>0:
                #     print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            if epoch % 1 == 0:
                self.fast_evaluation(epoch)
            print('each epoch: {} seconds'.format(time.time() - t0))
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        user_emb = self.user_emb[u]
        item_emb = self.item_emb
        if args.loss in ['directau']:
            user_emb = F.normalize(user_emb, dim=-1)
            item_emb = F.normalize(item_emb, dim=-1)

        score = torch.matmul(user_emb, item_emb.transpose(0, 1))
        return score.cpu().numpy()

class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

        self.fkan_layer = fJNB(degree=1)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict


    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        # 应用 FKAN 层
        #ego_embeddings = self.fkan_layer(ego_embeddings)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            # 应用 FKAN 层
            ego_embeddings = self.fkan_layer(ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        # 应用 FKAN 层
        #all_embeddings = self.fkan_layer(all_embeddings)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        # 应用 FKAN 层
        #user_all_embeddings = self.fkan_layer(user_all_embeddings)
        #item_all_embeddings = self.fkan_layer(item_all_embeddings)


        return user_all_embeddings, item_all_embeddings



#  2  添加CaNet
import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
# from util.loss_torch import bpr_loss,l2_reg_loss, bpr_k_loss, ccl_loss, directau_loss, gbce_loss, simce_loss, ssm_loss
from util.loss_torch import bpr_loss, l2_reg_loss, bpr_k_loss, ccl_loss, directau_loss, simce_loss, ssm_loss
import torch.nn.functional as F
from fkan.torch import FractionalJacobiNeuralBlock as fJNB
import numpy as np
from torch_geometric.nn import GCNConv

from util.args import get_params
import time

args = get_params()


class EnvironmentEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, env_dim):
        super(EnvironmentEncoder, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, env_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        return x


class LightGCN(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(LightGCN, self).__init__(conf, training_set, test_set)
        _args = OptionConf(self.config['LightGCN'])
        self.n_layers = int(_args['-n_layer'])

        self.env_dim = 16  # 可以根据需要修改
        self.sequence_length = 10  # 用户短期交互序列长度，可以调整

        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers, self.env_dim, self.sequence_length)
        if args.loss in ['ssm', 'simce']:
            self.maxEpoch = 50

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        for epoch in range(self.maxEpoch):
            t0 = time.time()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):

                user_idx, pos_idx, neg_idx = batch

                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                    neg_idx]

                reg_loss = l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb) / self.batch_size

                if args.loss == 'bpr':
                    rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                elif args.loss == 'ssm':
                    rec_loss = ssm_loss(user_emb, pos_item_emb, neg_item_emb)
                elif args.loss == 'simce':
                    rec_loss = simce_loss(user_emb, pos_item_emb, neg_item_emb, margin=args.margin)

                batch_loss = rec_loss + reg_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                # if n % 100==0 and n>0:
                #     print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            if epoch % 1 == 0:
                self.fast_evaluation(epoch)
            print('each epoch: {} seconds'.format(time.time() - t0))
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        user_emb = self.user_emb[u]
        item_emb = self.item_emb
        if args.loss in ['directau']:
            user_emb = F.normalize(user_emb, dim=-1)
            item_emb = F.normalize(item_emb, dim=-1)

        score = torch.matmul(user_emb, item_emb.transpose(0, 1))
        return score.cpu().numpy()


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, env_dim, sequence_length):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.sequence_length = sequence_length

        self.fkan_layer = fJNB(degree=1)
        # 创建物品ID到数值ID的映射
        self.item_id_map = {item: idx for idx, item in self.data.id2item.items()}
        # ----  实例化环境估计器  ----
        self.env_encoder = EnvironmentEncoder(emb_size, 64, env_dim).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)

        # ----  获取环境信息  ----
        user_ids = []
        item_ids = []
        for user in self.data.training_set_u:
            for item in self.data.training_set_u[user]:
                user_ids.append(self.data.user[user])
                item_ids.append(self.item_id_map[item])

        user_ids = np.array(user_ids)
        item_ids = np.array(item_ids)
        edge_index = torch.tensor([user_ids, item_ids], dtype=torch.long).cuda()

        user_sequences = []
        for user in self.data.training_set_u:  # 遍历所有用户
            user_id = self.data.user[user]
            user_interactions = list(self.data.training_set_u[user].keys())
            if len(user_interactions) >= self.sequence_length:
                # 将物品ID转换为数值ID
                user_sequences.append([self.item_id_map[i] for i in user_interactions[-self.sequence_length:]])
            else:
                padding = [0] * (self.sequence_length - len(user_interactions))
                # 将物品ID转换为数值ID
                user_sequences.append(padding + [self.item_id_map[i] for i in user_interactions])

        user_sequences = torch.tensor(user_sequences).long().cuda()  # 转换为tensor
        sequence_embeddings = self.embedding_dict['item_emb'][user_sequences]  # 获取序列中物品的embedding
        sequence_embeddings = torch.mean(sequence_embeddings, dim=1)  # 对序列embedding取平均
        env_embeddings = self.env_encoder(ego_embeddings, edge_index)

        # 将环境表示与用户短期交互表示拼接
        user_env_embeddings = torch.cat([env_embeddings[:self.data.user_num], sequence_embeddings], dim=1)
        item_env_embeddings = torch.cat([torch.zeros(self.data.item_num, self.latent_size).cuda(),
                                         env_embeddings[self.data.user_num:]], dim=1)
        env_embeddings = torch.cat([user_env_embeddings, item_env_embeddings], dim=0)

        # 应用 FKAN 层
        # ego_embeddings = self.fkan_layer(ego_embeddings)
        # ---- 进行消息传递 ----
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            # 将环境表示与节点表示拼接
            ego_embeddings = torch.cat([ego_embeddings, env_embeddings], dim=-1)
            # 进行消息传递
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            # 将节点嵌入的维度恢复到原始维度
            ego_embeddings = ego_embeddings[:, :self.latent_size]

            # 应用 FKAN 层
            # ego_embeddings = self.fkan_layer(ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        # 应用 FKAN 层
        # all_embeddings = self.fkan_layer(all_embeddings)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        # 应用 FKAN 层
        # user_all_embeddings = self.fkan_layer(user_all_embeddings)
        # item_all_embeddings = self.fkan_layer(item_all_embeddings)

        return user_all_embeddings, item_all_embeddings



#3  fkan+care
import os
import pickle

import torch
import torch.nn as nn
from mlxtend.preprocessing import TransactionEncoder

from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, simce_loss, ssm_loss
import torch.nn.functional as F
from fkan.torch import FractionalJacobiNeuralBlock as fJNB

from util.args import get_params
import time
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.frequent_patterns import association_rules
import pandas as pd

args = get_params()


class LightGCN(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(LightGCN, self).__init__(conf, training_set, test_set)
        _args = OptionConf(self.config['LightGCN'])
        self.n_layers = int(_args['-n_layer'])
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)
        if args.loss in ['ssm', 'simce']:
            self.maxEpoch = 50

        # ===  新增参数：用于因果正则化的关联规则  ===
        self.rules = self.mine_association_rules(self.data.interaction_mat.tocsr(), dataset_name="amazon-cd")
        # ===  新增参数：一致性损失权重  ===
        self.consistency_weight = 0.01  # 可以根据需要调整

        # ===  新增参数：规则数量限制  ===
        self.max_rules = 10000  # 可以根据需要调整

        # ===  新增参数：采样比例  ===
        self.sample_ratio = 0.1  # 可以根据需要调整

    def mine_association_rules(self, interaction_mat, dataset_name, rules_dir="rules"):
        """
        使用关联规则挖掘，从交互矩阵中挖掘频繁的物品组合.

        参数:
            interaction_mat: 用户-物品交互矩阵，类型为 scipy.sparse.csr_matrix

        返回值:
            rules: DataFrame, 包含关联规则，格式为 ['antecedents', 'consequents', 'confidence']
        """
        rules_file = os.path.join(rules_dir, f"{dataset_name}_rules.pkl")
        if os.path.exists(rules_file):
            with open(rules_file, 'rb') as f:
                rules = pickle.load(f)
            print(f"关联规则从文件 '{rules_file}' 中加载.")
        else:
            # 将交互矩阵转换为适合 Apriori 算法的格式
            transactions = interaction_mat.tolil().rows
            te = TransactionEncoder()  # 创建 TransactionEncoder 对象
            te_ary = te.fit(transactions).transform(transactions)  # 进行转换
            transactions_df = pd.DataFrame(te_ary, columns=te.columns_)  # 转换为 DataFrame

            # 使用 fpgrowth 算法挖掘频繁项集
            # 根据数据调整 min_support
            frequent_itemsets = fpgrowth(transactions_df, min_support=0.01, use_colnames=True)

            # 生成关联规则
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)  # 根据数据调整 min_threshold
            os.makedirs(rules_dir, exist_ok=True)  # 创建目录 (如果不存在)
            with open(rules_file, 'wb') as f:
                pickle.dump(rules, f)
            print(f"关联规则保存到文件 '{rules_file}'.")
        return rules

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            t0 = time.time()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):

                user_idx, pos_idx, neg_idx = batch

                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                    neg_idx]

                # ===  计算推荐损失  ===
                reg_loss = l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb) / self.batch_size
                if args.loss == 'bpr':
                    rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                elif args.loss == 'ssm':
                    rec_loss = ssm_loss(user_emb, pos_item_emb, neg_item_emb)
                elif args.loss == 'simce':
                    rec_loss = simce_loss(user_emb, pos_item_emb, neg_item_emb, margin=args.margin)
                else:
                    raise ValueError(f"Unsupported loss type: {args.loss}")

                # ===  计算一致性损失  ===
                if len(self.rules) > self.max_rules:
                    rules = self.rules.sample(n=self.max_rules)
                else:
                    rules = self.rules
                sampled_rules = rules.sample(frac=self.sample_ratio)  # 从已限制数量的规则中采样
                consistency_loss = self.calculate_consistency_loss(rec_user_emb, sampled_rules)

                # ===  组合损失  ===
                batch_loss = rec_loss + reg_loss + self.consistency_weight * consistency_loss

                # ===  反向传播和优化  ===
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            if epoch % 1 == 0:
                self.fast_evaluation(epoch)
            print('each epoch: {} seconds'.format(time.time() - t0))
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def calculate_consistency_loss(self, user_embeddings, rules, temperature=0.3):
        """
        计算一致性损失。

        参数:
            user_embeddings: 用户嵌入矩阵，形状为 (num_users, embedding_dim)
            rules: DataFrame, 包含关联规则，格式为 ['antecedents', 'consequents', 'confidence']
            temperature: 温度参数，用于控制相似度分布的平滑程度

        返回值:
            loss: 一致性损失值
        """
        loss = 0
        for _, rule in rules.iterrows():
            antecedent = int(list(rule['antecedents'])[0])  # 获取原因物品
            consequent = int(list(rule['consequents'])[0])  # 获取结果物品

            # 获取购买了 antecedent 和 consequent 物品的用户的索引
            antecedent_users = self.data.interaction_mat[:, antecedent].nonzero()[0]
            consequent_users = self.data.interaction_mat[:, consequent].nonzero()[0]

            # 计算购买了 antecedent 物品的用户的平均嵌入
            if antecedent_users.size > 0:
                antecedent_emb = user_embeddings[antecedent_users].mean(dim=0)
            else:
                continue

            # 计算购买了 consequent 物品的用户的平均嵌入
            if consequent_users.size > 0:
                consequent_emb = user_embeddings[consequent_users].mean(dim=0)
            else:
                continue

            # 使用余弦相似度计算一致性损失
            loss += 1 - F.cosine_similarity(antecedent_emb.unsqueeze(0), consequent_emb.unsqueeze(0))

        return loss

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        user_emb = self.user_emb[u]
        item_emb = self.item_emb
        if args.loss in ['directau']:
            user_emb = F.normalize(user_emb, dim=-1)
            item_emb = F.normalize(item_emb, dim=-1)

        score = torch.matmul(user_emb, item_emb.transpose(0, 1))
        return score.cpu().numpy()

class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

        self.fkan_layer = fJNB(degree=1)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict


    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        # 应用 FKAN 层
        #ego_embeddings = self.fkan_layer(ego_embeddings)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            # 应用 FKAN 层
            #ego_embeddings = self.fkan_layer(ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        # 应用 FKAN 层
        #all_embeddings = self.fkan_layer(all_embeddings)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        # 应用 FKAN 层
        #user_all_embeddings = self.fkan_layer(user_all_embeddings)
        #item_all_embeddings = self.fkan_layer(item_all_embeddings)


        return user_all_embeddings, item_all_embeddings
