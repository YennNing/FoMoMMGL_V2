import torch
from transformers import AutoTokenizer
import pickle
import pandas as pd
from PIL import Image
from urllib.request import urlopen

from language_modelling import utils

def load_wikiweb2m_tf(task):
    with open(f'./wikiweb2m/raw/wikiweb2m_{task}_train_medium.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    with open(f'./wikiweb2m/raw/wikiweb2m_{task}_val_medium.pkl', 'rb') as f:
        val_dataset = pickle.load(f)
    with open(f'./wikiweb2m/raw/wikiweb2m_{task}_test_medium.pkl', 'rb') as f:
        test_dataset = pickle.load(f)

    return train_dataset, val_dataset[:10000], test_dataset[:10000]

def load_wikiweb2m(task):
    train_df = pd.read_parquet(f'./wikiweb2m/raw/wikiweb2m_train.parquet')
    val_df = pd.read_parquet(f'./wikiweb2m/raw/wikiweb2m_val.parquet')
    test_df = pd.read_parquet(f'./wikiweb2m/raw/wikiweb2m_test.parquet')

    with open(f'./wikiweb2m/raw/{task}_id_split.pkl', 'rb') as f:
        id_list = pickle.load(f)

    return train_df, val_df, test_df, id_list


class WikiWeb2M(torch.utils.data.Dataset):

    def __init__(self, args, df, id_list, tokenizer, feature_extractor_model=None):
        self.path = './wikiweb2m/raw/'
        self.task = args.task
        self.context = args.context

        self.df = df
        self.id_list = id_list
        self.tokenizer = tokenizer
        self.max_input_length = args.max_input_length
        self.max_output_length = args.max_output_length

        if feature_extractor_model is not None and self.context == 'all':
            self.feature_extractor = utils.get_feature_extractor_for_model(feature_extractor_model)

    def __len__(self):
        return len(self.id_list)

    def get_page_info(self, d):
        page_url = d[0]['page_url'].numpy().decode()
        page_title = d[0]['page_title'].numpy().decode()
        page_description = d[0]['clean_page_description'].numpy().decode()
        #return ", ".join([page_url, page_title, page_description])
        return ", ".join([page_title, page_description])

    def get_section_info(self, section_id, d, remove_summary=True):
        page_title = d['page_title'].decode()
        section_title = d['section_title'][section_id].decode()
        section_depth = str(d['section_depth'][section_id])
        section_heading = str(d['section_heading'][section_id])
        section_parent_index = str(d['section_parent_index'][section_id])
        section_summary = d['section_summary'][section_id].decode()
        section_rest_sentence = d['section_rest_sentence'][section_id].decode()
        if remove_summary:
            #return ", ".join([section_title, section_depth, section_heading, section_parent_index, section_rest_sentence]), section_summary
            return ", ".join([section_rest_sentence]), section_summary
        else:
            #return ", ".join([section_title, section_depth, section_heading, section_parent_index, section_summary, section_rest_sentence])
            return ", ".join([section_summary, section_rest_sentence])

    def get_section_images(self, section_id, d, omit_image_id=-1):
        image_urls =  tf.sparse.to_dense(d[1]['section_image_url'])
        section_image_info = []
        for image_id in range(image_urls[section_id].shape[0]):
            if image_urls[section_id][image_id].numpy() == b'':
                continue
            if image_id == omit_image_id:
                continue
            image_url = image_urls[section_id][image_id].numpy().decode()
            image_caption = tf.sparse.to_dense(d[1]['section_image_captions'])[section_id][image_id].numpy().decode()

            img = Image.open(urlopen(image_url))
            image = utils.get_pixel_values_for_model(self.feature_extractor, img)
            section_image_info.append((image, image_caption))
        return section_image_info

    def get_image_info(self, section_id, image_id, d, remove_caption=True):
        image_url =  tf.sparse.to_dense(d[1]['section_image_url'])[section_id][image_id].numpy().decode()
        image_caption = tf.sparse.to_dense(d[1]['section_image_captions'])[section_id][image_id].numpy().decode()
        if remove_caption:
            return Image.open(urlopen(image_url))
        else:
            return Image.open(urlopen(image_url)), image_caption

    def __getitem__(self, index):
        if self.task == "section":
            page_id, section_id = self.id_list[index]
            d = self.df[self.df['page_id'] == page_id].iloc[0]
            if self.context == "section_only":
                section_info, labels = self.get_section_info(section_id, d, remove_summary=True)
                inputs = "summarize: " + section_info
            elif self.context == "text_only":
                page_info = self.get_page_info(d)
                section_info, labels = self.get_section_info(section_id, d, remove_summary=True)
                context_num = tf.sparse.to_dense(d[1]['section_title']).shape[0]
                context_info = []
                for context_id in range(context_num):
                    if context_id == section_id:
                        continue
                    context_info.append(self.get_section_info(context_id, d, remove_summary=False))
                context_info = ', '.join(context_info)
                inputs = "summarize: " + section_info + ", context: " + page_info + context_info
            elif self.context == "all":
                page_info = self.get_page_info(d)
                section_info, labels = self.get_section_info(section_id, d, remove_summary=True)
                section_image_info = self.get_section_images(section_id, d)
                context_num = tf.sparse.to_dense(d[1]['section_title']).shape[0]
                context_info = []
                for context_id in range(context_num):
                    if context_id == section_id:
                        continue
                    context_text = self.get_section_info(context_id, d, remove_summary=False)
                    context_image = self.get_section_images(context_id, d)
                    context_info.append((context_text, context_info))
                context_info = ', '.join(context_info)
                inputs = "summarize: " + section_info + ", context: " + page_info + context_info

        inputs, labels = inputs.replace('\n', ''), labels.replace('\n', '')
        inputs, labels = ' '.join(inputs.split()), ' '.join(labels.split())

        # Tokenize
        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.tokenizer(labels, max_length=self.max_output_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
        labels_with_ignore_index = [label if label != 0 else -100 for label in labels[0]]

        input_ids = model_inputs.input_ids[0]
        attention_mask = model_inputs.attention_mask[0]
        labels = torch.LongTensor(labels_with_ignore_index)

        return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
                }

    @torch.no_grad()
    def collate(self, items):
        (inputs, labels) = zip(*[(item["inputs"], item["labels"]) for item in items])
        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.tokenizer(labels, max_length=self.max_output_length, padding="max_length", truncation=True, return_tensors="pt").input_ids

        labels_with_ignore_index = []
        for example in labels:
            labels_with_ignore_index.append([label if label != 0 else -100 for label in example])

        return {
                "input_ids": torch.LongTensor(model_inputs.input_ids),
                "attention_mask": torch.LongTensor(model_inputs.attention_mask),
                "labels": torch.LongTensor(labels_with_ignore_index)
                }

