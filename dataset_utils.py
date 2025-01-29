import os

import pandas as pd
from torch.utils.data import Dataset, DataLoader


class NLIClassificationDataset(Dataset):
    def __init__(self,
                 data, 
                 prompt_building_function, # how to build the prompt
                 indexing_function, # how to get index from data
                 label_transforming_function # transform the lable to the correct one
                ):
        super().__init__()
        self.data = data
        self.prompt_building_function = prompt_building_function
        self.indexing_function = indexing_function
        self.label_transforming_function = label_transforming_function

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.prompt_building_function(self.indexing_function(self.data, index), self.label_transforming_function)


class NLIPreprocessorUtils:
    
    @staticmethod
    def jsonl_dataset_to_df(path):
        with open(path, 'r') as f:
            l = []
            for line in f:
                l.append(eval(line))
        return pd.DataFrame(l)

    '''
    candidate prompt building functions
    '''
    
    @staticmethod
    def mnli_classifier_prompt_build(row, label_transforming_function):
        '''
        Build prompts in the form (prompt, label) with label_transforming_function for MNLI
        '''
        return f"Please indicate the relationship between the premise and the hypothesis with entailment, neutral or contradiction. Premise: {row['sentence1']} Hypothesis: {row['sentence2']} The relationship between premise and hypothesis is", f"{label_transforming_function(str(row['gold_label']))}"

    def hans_classifier_prompt_build(row, label_transform_func):
        '''
        Build prompts in the form (prompt, label) with label_transforming_function for HANS
        '''
        return f"Please indicate the relationship between the premise and the hypothesis with entailment, neutral or contradiction. Premise: {row['sentence1']} Hypothesis: {row['sentence2']} The relationship between premise and hypothesis is", f"{label_transform_func(str(row['gold_label']))}"

    
    '''
    candidate label transforming functions
    '''

    @staticmethod
    def label_mili_3_to_hans_2(label):
        mnli_label_transform_dict = {
            'entailment': 'entailment',
            'neutral': 'non-entailment',
            'contradiction': 'non-entailment'
        }
        return mnli_label_transform_dict[label]


class NLIPreprocessor:
    def __init__(
        self,
        label_transforming_function=lambda x: x,
        prompt_building_function=NLIPreprocessorUtils.mnli_classifier_prompt_build
    ):
        self.label_transforming_function = label_transforming_function
        self.prompt_building_function = prompt_building_function

class MNLIPreprocessor(NLIPreprocessor):
    def __init__(self, 
                 mnli_path,
                 label_encoder,
                 validation_option='balance',
                 validation_size_per_category=250,
                 label_transforming_function=lambda x: x,
                 prompt_building_function=NLIPreprocessorUtils.mnli_classifier_prompt_build,
                ):
        super().__init__(label_transforming_function, prompt_building_function)
        
        self.mnli_path = mnli_path

        self.label_encoder = label_encoder

        self.validation_option = validation_option
        self.validation_size_per_category = validation_size_per_category

        # generate dataframes
        self.original_train_df = NLIPreprocessorUtils.jsonl_dataset_to_df(os.path.join(mnli_path, 'multinli_1.0_train.jsonl')).query('gold_label != "-"').drop_duplicates(['sentence1', 'sentence2'], keep=False)
        self.dev_matched_df = NLIPreprocessorUtils.jsonl_dataset_to_df(os.path.join(mnli_path, 'multinli_1.0_dev_matched.jsonl')).query('gold_label != "-"').drop_duplicates(['sentence1', 'sentence2'], keep=False)
        self.dev_mismatched_df = NLIPreprocessorUtils.jsonl_dataset_to_df(os.path.join(mnli_path, 'multinli_1.0_dev_mismatched.jsonl')).query('gold_label != "-"').drop_duplicates(['sentence1', 'sentence2'], keep=False)
        
        # generate datasets
        self.original_train_dataset = NLIClassificationDataset(
            data=self.original_train_df,
            prompt_building_function=self.prompt_building_function,
            indexing_function=lambda data, index: data.iloc[index],
            label_transforming_function=self.label_transforming_function
        )

        self.dev_matched_dataset = NLIClassificationDataset(
            data=self.dev_matched_df,
            prompt_building_function=self.prompt_building_function,
            indexing_function=lambda data, index: data.iloc[index],
            label_transforming_function=self.label_transforming_function
        )

        self.dev_mismatched_dataset = NLIClassificationDataset(
            data=self.dev_mismatched_df,
            prompt_building_function=self.prompt_building_function,
            indexing_function=lambda data, index: data.iloc[index],
            label_transforming_function=self.label_transforming_function
        )

        # special datasets used for training
        if validation_option == 'balance':
            # same size for each label
            self.validation_df = pd.concat([
                self.original_train_df.query('gold_label == "contradiction"').iloc[:self.validation_size_per_category],
                self.original_train_df.query('gold_label == "neutral"').iloc[:self.validation_size_per_category],
                self.original_train_df.query('gold_label == "entailment"').iloc[:self.validation_size_per_category],
            ])

        elif validation_option == 'balance_reduced':
            # same size for entailment and non-entailment
            self.validation_df = pd.concat([
                self.original_train_df.query('gold_label == "contradiction"').iloc[:self.validation_size_per_category],
                self.original_train_df.query('gold_label == "neutral"').iloc[:self.validation_size_per_category],
                self.original_train_df.query('gold_label == "entailment"').iloc[:self.validation_size_per_category * 2],
            ])
        
        # take the rest of the training set
        self.train_df = pd.concat([
            self.original_train_df,
            self.validation_df
        ]).drop_duplicates(subset=['sentence1', 'sentence2'], keep=False)

        # generate datasets of train and validation
        self.train_dataset = NLIClassificationDataset(
            data=self.train_df,
            prompt_building_function=self.prompt_building_function,
            indexing_function=lambda data, index: data.iloc[index],
            label_transforming_function=self.label_transforming_function
        )

        self.validation_dataset = NLIClassificationDataset(
            data=self.validation_df,
            prompt_building_function=self.prompt_building_function,
            indexing_function=lambda data, index: data.iloc[index],
            label_transforming_function=self.label_transforming_function
        )

    def mnli_dataset_tokenize(self, dataset, tokenizer):
        return [{**tokenizer(x[0]), **{'label': self.label_encoder(x[1])}} for x in dataset]

    def get_sampled_train_dataset(self, n=None, frac=None, random_state=None):
        sampled_train_df = self.train_df.sample(n=n, frac=frac, random_state=random_state)
        ret_dataset = NLIClassificationDataset(
            data=sampled_train_df,
            prompt_building_function=self.prompt_building_function,
            indexing_function=lambda data, index: data.iloc[index],
            label_transforming_function=self.label_transforming_function
        )
        return ret_dataset

    def combine_text_and_label(self, item, label_first=False):
        if not label_first:
            return f"{item[0]}: {item[1]}."
        else:
            return f"This is an example where the relationship between the premise and the hypothesis is {item[1]}. {item[0][115:-51]}"


class HANSPreprocessor(NLIPreprocessor):
    def __init__(self, 
                 hans_path, 
                 label_encoder,
                 mini_set_per_subcase_size=20,
                 label_transforming_function=lambda x: x,
                 prompt_building_function=NLIPreprocessorUtils.hans_classifier_prompt_build
                ):
        super().__init__(label_transforming_function, prompt_building_function)
        self.hans_path = hans_path

        self.label_encoder = label_encoder

        self.mini_set_per_subcase_size = mini_set_per_subcase_size

        self.train_df = pd.DataFrame(NLIPreprocessorUtils.jsonl_dataset_to_df(os.path.join(hans_path, 'heuristics_train_set.jsonl')))
        self.validation_df = pd.DataFrame(pd.DataFrame(NLIPreprocessorUtils.jsonl_dataset_to_df(os.path.join(hans_path, 'heuristics_evaluation_set.jsonl'))))

        self.train_mini_df = pd.concat([x.iloc[:self.mini_set_per_subcase_size] for _, x in self.train_df.groupby('subcase')])
        self.validation_mini_df = pd.concat([x.iloc[:self.mini_set_per_subcase_size] for _, x in self.validation_df.groupby('subcase')])

        # generate datasets
        self.train_dataset = NLIClassificationDataset(
            data=self.train_df,
            prompt_building_function=self.prompt_building_function,
            indexing_function=lambda data, index: data.iloc[index],
            label_transforming_function=self.label_transforming_function
        )

        self.validation_dataset = NLIClassificationDataset(
            data=self.validation_df,
            prompt_building_function=self.prompt_building_function,
            indexing_function=lambda data, index: data.iloc[index],
            label_transforming_function=self.label_transforming_function
        )

        self.train_mini_dataset = NLIClassificationDataset(
            data=self.train_mini_df,
            prompt_building_function=self.prompt_building_function,
            indexing_function=lambda data, index: data.iloc[index],
            label_transforming_function=self.label_transforming_function
        )

        self.validation_mini_dataset = NLIClassificationDataset(
            data=self.validation_mini_df,
            prompt_building_function=self.prompt_building_function,
            indexing_function=lambda data, index: data.iloc[index],
            label_transforming_function=self.label_transforming_function
        )

        # by heuristic dataframes
        self.heuristic_list = self.train_df['heuristic'].drop_duplicates().to_list()
        self.train_df_by_heuristic = {}
        self.validation_df_by_heuristic = {}
        for heur in self.heuristic_list:
            self.train_df_by_heuristic[heur] = self.train_df.query(f'heuristic == "{heur}"')
            self.validation_df_by_heuristic[heur] = self.validation_df.query(f'heuristic == "{heur}"')

        # by subcase dataframes
        self.subcase_list = self.train_df['subcase'].drop_duplicates().to_list()
        self.train_df_by_subcase = {}
        self.validation_df_by_subcase = {}
        for sc in self.subcase_list:
            self.train_df_by_subcase[sc] = self.train_df.query(f'subcase == "{sc}"')
            self.validation_df_by_subcase[sc] = self.validation_df.query(f'subcase == "{sc}"')

        # subdatasets
        self.train_dataset_by_heuristic = {}
        self.validation_dataset_by_heuristic = {}
        for heur in self.heuristic_list:
            self.train_dataset_by_heuristic[heur] = NLIClassificationDataset(
                data=self.train_df_by_heuristic[heur],
                prompt_building_function=self.prompt_building_function,
                indexing_function=lambda data, index: data.iloc[index],
                label_transforming_function=self.label_transforming_function
            )
            self.validation_dataset_by_heuristic[heur] = NLIClassificationDataset(
                data=self.validation_df_by_heuristic[heur],
                prompt_building_function=self.prompt_building_function,
                indexing_function=lambda data, index: data.iloc[index],
                label_transforming_function=self.label_transforming_function
            )

        self.train_dataset_by_subcase = {}
        self.validation_dataset_by_subcase = {}
        for sc in self.subcase_list:
            self.train_dataset_by_subcase[sc] = NLIClassificationDataset(
                data=self.train_df_by_subcase[sc],
                prompt_building_function=self.prompt_building_function,
                indexing_function=lambda data, index: data.iloc[index],
                label_transforming_function=self.label_transforming_function
            )
            self.validation_dataset_by_subcase[sc] = NLIClassificationDataset(
                data=self.validation_df_by_subcase[sc],
                prompt_building_function=self.prompt_building_function,
                indexing_function=lambda data, index: data.iloc[index],
                label_transforming_function=self.label_transforming_function
            )

    def hans_dataset_tokenize(self, dataset, tokenizer):
        return [{**tokenizer(x[0]), **{'label': self.label_encoder(x[1])}} for x in dataset]

class ClassificationLablesEncoder(object):
    def __init__(self, all_labels):
        self.label_dict = {l: i for i, l in enumerate(all_labels)}

    def __call__(self, label):
        return self.label_dict[label]


