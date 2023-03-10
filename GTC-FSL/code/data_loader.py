
from torch.utils.data import Sampler
from typing import List, Iterable
import numpy as np
import torch
import pandas as pd

from torch.utils.data import Dataset
import json
from tqdm import tqdm


def get_data(path, mode='json'):
    result = []
    with open(path, 'r') as src:
        if mode == 'json':
            for line in tqdm(src):
                line = json.loads(line)
                result.append(line)
        else:
            for line in tqdm(src):
                line = line.split('\n')[0]
                result.append(line)
    return result
        
def write_data(datas, path):
    with open(path, 'w') as fout:
        for line in datas:
            fout.write("%s\n" % json.dumps(line, ensure_ascii=False))

class KShotTaskSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 episodes_per_epoch: int = None,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 num_tasks: int = 1,
                 fixed_tasks: List[Iterable[int]] = None):
        """PyTorch Sampler subclass that generates batches of n-shot, k-way, q-query tasks.

        Each n-shot task contains a "support set" of `k` sets of `n` samples and a "query set" of `k` sets
        of `q` samples. The support set and the query set are all grouped into one Tensor such that the first n * k
        samples are from the support set while the remaining q * k samples are from the query set.

        The support and query sets are sampled such that they are disjoint i.e. do not contain overlapping samples.

        # Arguments
            dataset: Instance of torch.utils.data.Dataset from which to draw samples
            episodes_per_epoch: Arbitrary number of batches of n-shot tasks to generate in one epoch
            n_shot: int. Number of samples for each class in the n-shot classification tasks.
            k_way: int. Number of classes in the n-shot classification tasks.
            q_queries: int. Number query samples for each class in the n-shot classification tasks.
            num_tasks: Number of n-shot tasks to group into a single batch
            fixed_tasks: If this argument is specified this Sampler will always generate tasks from
                the specified classes
        """
        super(KShotTaskSampler, self).__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset
        if num_tasks < 1:
            raise ValueError('num_tasks must be > 1.')

        self.num_tasks = num_tasks
        # TODO: Raise errors if initialise badly
        self.k = k
        self.n = n
        self.q = q
        self.fixed_tasks = fixed_tasks

        self.i_task = 0

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            support_set, query_set = [], []

            for task in range(self.num_tasks):
                if self.fixed_tasks is None:
                    # Get random classes
                    episode_classes = np.random.choice(self.dataset.df['class_id'].unique(), size=self.k, replace=False)
                else:
                    # Loop through classes in fixed_tasks
                    episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1

                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]

                support_k = {k: None for k in episode_classes}
                episode_labels = []
                for k in episode_classes:
                    # Select support examples
                    support = df[df['class_id'] == k].sample(self.n)
                    support_k[k] = support
                    for i, s in support.iterrows():
                        tmp = {"text": s['text'], "label": s['class_name']}
                        support_set.append(tmp)
                    # if len(support) == 0:
                    #     episode_labels.append(k)
                    if len(support) != 0:
                        episode_labels.append(s['class_name'])
                q_set = []
                for m in range(0,10):
                    query_set = []
                    if len(support) == 0:
                        episode_labels = []
                    for k in episode_classes:
                        if len(df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))])<self.q:
                            query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].sample(self.q, replace = True)
                        else:
                            query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].sample(self.q)
                        for i, q in query.iterrows():
                            tmp = {"text": q['text'], "label": q['class_name']}
                            query_set.append(tmp)
                        if len(support) == 0:
                            episode_labels.append(q['class_name'])
                    q_set.append(np.stack(query_set))
                if len(support_set) == 0:
                    s = 0
                else:
                    s = np.stack(support_set)
            yield s, q_set, episode_labels


class MyDataset(Dataset):
    def __init__(self, path):
        """Dataset class representing FewAsp dataset
        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        

        self.df = pd.DataFrame(self.index_subset(path))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
       
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def __getitem__(self, item):
        label = self.datasetid_to_class_id[item]
        text = self.df['text'][item]

        return text, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(path):
        """Index a subset by looping through all of its files and recording relevant information.
        # Arguments
            subset: Name of the subset
        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            Omniglot dataset dataset
        """
        texts = []
        print('Indexing {}...'.format(path))
        
        datas = get_data(path)
        for line in tqdm(datas):      
            texts.append({
                    'text': line["text"],
                    'class_name': line["label"]
            })
        return texts



# test case
# train_dataset = MyDataset("../train.json")

# sampler = KShotTaskSampler(train_dataset, episodes_per_epoch=2, n=2, k=2, q=1, num_tasks=2)
# for i, batch in enumerate(sampler):
   
#     support, query, classes = batch
#     print(i, len(support), len(query))
#     print(len(classes))
#     print("-"*50)
#     print(support)
#     print("-"*50)
#     print(query)
    
   