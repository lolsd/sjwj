import pandas as pd
import json
import os
data_path = "wine-reviews"
data_file_list = ["winemag-data_first150k.csv","winemag-data-130k-v2.csv"]
# data_path = "oakland-crime-statistics-2011-to-2016"
# data_file_list = ["records-for-2011.csv", "records-for-2012.csv", "records-for-2013.csv", "records-for-2014.csv", "records-for-2015.csv", "records-for-2016.csv"]

min_support = 0.25
min_confidence = 0.5

def makedir(path):
	if not os.path.exists(path):
		os.makedirs(path)

class Data():
    def __init__(self):
        self.read_data = os.path.join("result", data_path)
        self.write_data = os.path.join("result", data_path)
        self.data_file_list = data_file_list
    
    def association(self):
        for file_name in self.data_file_list:
            content = pd.read_csv(os.path.join(self.read_data, file_name.split('.')[0], 'processd.csv'))
            write_data_path = os.path.join(self.write_data, file_name.split('.')[0])
            makedir(write_data_path)
            print("--------------------------------------------------------------------------------------------------------------")
            print("Begin to process file: %s" % file_name)

            data_set = self.get_data_set(content)
            # 获取频繁项集
            freq_set, support_data = self.apriori(data_set)
            support_data_out = sorted(support_data.items(), key= lambda d:d[1],reverse=True)
            # 将频繁项集输出到结果文件
            freq_set_file = open(os.path.join(write_data_path, 'freq_set.json'), 'w')
            for (key, value) in support_data_out:
                result_dict = {'set':None, 'sup':None}
                set_result = list(key)
                sup_result = value
                result_dict['set'] = set_result
                result_dict['sup'] = sup_result
                json_str = json.dumps(result_dict, ensure_ascii=False)
                freq_set_file.write(json_str+'\n')
            freq_set_file.close()

            # 获取强关联规则列表
            big_rules_list = self.generate_rules(freq_set, support_data)
            big_rules_list = sorted(big_rules_list, key= lambda x:x[3], reverse=True)
            # 将关联规则输出到结果文件
            rules_file = open(os.path.join(write_data_path, 'rules.json'), 'w')
            for result in big_rules_list:
                result_dict = {'X_set':None, 'Y_set':None, 'sup':None, 'conf':None, 'lift':None}
                X_set, Y_set, sup, conf, lift = result
                result_dict['X_set'] = list(X_set)
                result_dict['Y_set'] = list(Y_set)
                result_dict['sup'] = sup
                result_dict['conf'] = conf
                result_dict['lift'] = lift
                json_str = json.dumps(result_dict, ensure_ascii=False)
                rules_file.write(json_str + '\n')
            rules_file.close()
                
    def get_data_set(self, content):
        columns = []
        for title in content.columns.values:
            if title == "Unnamed: 0":
                    continue
            feature_col = [title] + list(content[title])
            columns.append(feature_col)
        rows = list(zip(*columns))
        dataset = []
        feature_names = rows[0]
        for data_line in rows[1:]:
            data_set = []
            for i, value in enumerate(data_line):
                data_set.append((feature_names[i], value))
            dataset.append(data_set)
        return dataset
    
    def apriori(self, dataset):
        C1 = self.create_C1(dataset)
        dataset = [set(data) for data in dataset]
        L1, support_data = self.scan_D(dataset, C1)
        L = [L1]
        k = 2
        while len(L[k-2]) > 0:
            Ck = self.apriori_gen(L[k-2], k)
            Lk, support_k = self.scan_D(dataset, Ck)
            support_data.update(support_k)
            L.append(Lk)
            k += 1
        return L, support_data

    def create_C1(self, dataset):
        # 扫描dataset，构建全部可能的单元素候选项集合(list)
        # 每个单元素候选项：（属性名，属性取值）
        C1 = []
        for data in dataset:
            for item in data:
                if [item] not in C1:
                    C1.append([item])
        C1.sort()
        return [frozenset(item) for item in C1]

    def scan_D(self, dataset, Ck):
        # 过滤函数
        # 根据待选项集Ck的情况，判断数据集D中Ck元素的出现频率
        # 过滤掉低于最小支持度的项集
        Ck_count = dict()
        for data in dataset:
            for cand in Ck:
                if cand.issubset(data):
                    if cand not in Ck_count:
                        Ck_count[cand] = 1
                    else:
                        Ck_count[cand] += 1

        num_items = float(len(dataset))
        return_list = []
        support_data = dict()
        # 过滤非频繁项集
        for key in Ck_count:
            support  = Ck_count[key] / num_items
            if support >= min_support:
                return_list.insert(0, key)
            support_data[key] = support
        return return_list, support_data
    
    def apriori_gen(self, Lk, k):
        # 当待选项集不是单个元素时， 如k>=2的情况下，合并元素时容易出现重复
        # 因此针对包含k个元素的频繁项集，对比每个频繁项集第k-2位是否一致
        return_list = []
        len_Lk = len(Lk)

        for i in range(len_Lk):
            for j in range(i+1, len_Lk):
                # 第k-2个项相同时，将两个集合合并
                L1 = list(Lk[i])[:k-2]
                L2 = list(Lk[j])[:k-2]
                L1.sort()
                L2.sort()
                if L1 == L2:
                    return_list.append(Lk[i] | Lk[j])
        return return_list
    
    def generate_rules(self, L, support_data):
        """
        产生强关联规则算法实现
        基于Apriori算法，首先从一个频繁项集开始，接着创建一个规则列表，
        其中规则右部只包含一个元素，然后对这些规则进行测试。
        接下来合并所有的剩余规则列表来创建一个新的规则列表，
        其中规则右部包含两个元素。这种方法称作分级法。
        :param L: 频繁项集
        :param support_data: 频繁项集对应的支持度
        :return: 强关联规则列表
        """
        big_rules_list = []
        for i in range(1, len(L)):
            for freq_set in L[i]:
                H1 = [frozenset([item]) for item in freq_set]
                # 只获取有两个或更多元素的集合
                if i > 1:
                    self.rules_from_conseq(freq_set, H1, support_data, big_rules_list)
                else:
                    self.cal_conf(freq_set, H1, support_data, big_rules_list)
        return big_rules_list
    
    def rules_from_conseq(self, freq_set, H, support_data, big_rules_list):
        # H->出现在规则右部的元素列表
        m = len(H[0])
        if len(freq_set) > (m+1):
            Hmp1 = self.apriori_gen(H, m+1)
            Hmp1 = self.cal_conf(freq_set, Hmp1, support_data, big_rules_list)
            if len(Hmp1) > 1:
                self.rules_from_conseq(freq_set, Hmp1, support_data, big_rules_list)

    def cal_conf(self, freq_set, H, support_data, big_rules_list):
        # 评估生成的规则
        prunedH = []
        for conseq in H:
            sup = support_data[freq_set]
            conf = sup / support_data[freq_set - conseq]
            lift = conf / support_data[freq_set - conseq]
            if conf >= min_confidence:
                big_rules_list.append((freq_set-conseq, conseq, sup, conf, lift))
                prunedH.append(conseq)
        return prunedH
    
if __name__ == '__main__':
    data = Data()

    data.association()