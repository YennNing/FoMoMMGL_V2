import os
import time
import torch
from transformers import AutoTokenizer
import pickle
import pandas as pd
from PIL import Image
from urllib.request import urlopen

from language_modelling import utils
from torch_geometric.data import Data


def load_wikiweb2m(task):
    train_df = pd.read_parquet(f'./wikiweb2m/raw/wikiweb2m_train_large_mini.parquet')
    val_df = pd.read_parquet(f'./wikiweb2m/raw/wikiweb2m_val_large_mini.parquet')
    test_df = pd.read_parquet(f'./wikiweb2m/raw/wikiweb2m_test_large_mini.parquet')

    with open(f'./wikiweb2m/raw/section_id_split_large_mini.pkl', 'rb') as f:
        id_list = pickle.load(f)

    return train_df, val_df, test_df, id_list


class WikiWeb2M(torch.utils.data.Dataset):

    def __init__(self, args, df, id_list, tokenizer):
        self.path = './wikiweb2m/raw/'
        self.image_path = f'{self.path}/images_mini/'
        if not os.path.exists(self.image_path) and args.context in ('section_all', 'all'):
            raise ValueError(f'{self.image_path} does not exist')

        self.task = args.task
        self.context = args.context
        self.decoder_only = args.decoder_only
        self.neighbor_mode = args.neighbor_mode

        self.max_text_neighbors = args.max_text_neighbors
        self.max_image_neighbors = args.max_image_neighbors
        self.position_type = args.position_type

        self.df = df
        self.id_list = id_list
        self.tokenizer = tokenizer
        self.max_input_length = args.max_input_length
        self.max_output_length = args.max_output_length
        self.text_model = args.text_model
        self.text_tokenizer = AutoTokenizer.from_pretrained(args.text_model, use_fast=False)
        self.text_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        if self.neighbor_mode in ('prefix', 'cross_attention'):
            self.text_model = args.text_model
            self.text_tokenizer = AutoTokenizer.from_pretrained(args.text_model, use_fast=False)
            self.text_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        if self.context in ('section_all', 'all'):
            self.visual_feature_extractor = utils.get_feature_extractor_for_model(args.visual_model)

        self.n_text_tokens = args.n_text_tokens
        self.n_visual_tokens = args.n_visual_tokens

    def __len__(self):
        return len(self.id_list)

    def get_page_info(self, d):
        page_url = d['page_url'].decode()
        page_title = d['page_title'].decode()
        page_description = d['page_description'].decode()
        page_info = ', '.join([page_title, page_description])
        return ' '.join(page_info.replace('\n', ' ').split())

    def get_section_info(self, section_id, d, remove_summary=True):
        section_depth = str(d['section_depth'][section_id])
        section_heading = str(d['section_heading'][section_id])
        section_parent_index = str(d['section_parent_index'][section_id])
        section_title = d['section_title'][section_id].decode()
        section_summary = d['section_summary'][section_id].decode()
        section_rest_sentence = d['section_rest_sentence'][section_id].decode()
        if remove_summary:
            section_info = ', '.join([section_rest_sentence])
            section_info, section_summary = ' '.join(section_info.replace('\n', ' ').split()), ' '.join(section_summary.replace('\n', ' ').split())
            return section_info, section_summary
        else:
            section_info = ', '.join([section_summary, section_rest_sentence])
            section_info = ' '.join(section_info.replace('\n', ' ').split())
            return section_info

    def get_section_images(self, page_id, section_id, d):
        section_num = d['section_title'].shape[0]
        image_urls = d['image_url'].reshape(section_num, -1)
        image_captions = d['image_caption'].reshape(section_num, -1)
        for image_id in range(image_urls[section_id].shape[0]):
            image_url = image_urls[section_id][image_id].decode()
            if image_url == '':
                continue
            file_format = os.path.splitext(image_url)[1][1:]
            file_name = f'{self.image_path}/{page_id}_{section_id}_{image_id}.{file_format}'
            if os.path.exists(file_name):
                try:
                    with Image.open(file_name) as img:
                        section_image = utils.get_pixel_values_for_model(self.visual_feature_extractor, img)
                        section_caption = image_captions[section_id][image_id].decode()
                except Exception as e:
                    print(f"Error encountered: {e}")
                    continue
                return section_image, ' '.join(section_caption.replace('\n', ' ').split())
        return None, None

    def __getitem__(self, index):
        if self.neighbor_mode in ("prefix", "cross_attention"):
            return self.get_embedding_item(index)

        page_id, section_id = self.id_list[index]
        d = self.df[self.df['page_id'] == page_id].iloc[0]
        if self.context == 'section_only':
            section_info, labels = self.get_section_info(section_id, d, remove_summary=True)
            inputs = 'summarize: ' + section_info
            input_ids = self.tokenizer(inputs, max_length=self.max_input_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0]

        elif self.context == "section_all":
            section_info, labels = self.get_section_info(section_id, d, remove_summary=True)
            image, image_caption = self.get_section_images(page_id, section_id, d)

            images = []
            image_positions = []
            if image is None:
                inputs = "summarize: " + section_info
                visual_ids = torch.LongTensor(self.n_visual_tokens * [self.tokenizer.pad_token_id])
                images.append(torch.zeros((3,  224, 224)))
            else:
                inputs = "summarize: " + section_info + ", context: " + image_caption
                visual_ids = torch.LongTensor(self.n_visual_tokens * [100])
                images.append(image)
            max_text_length = self.max_input_length - self.n_visual_tokens
            input_ids = self.tokenizer(inputs, max_length=max_text_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0]
            image_positions.append(input_ids.shape[0] + torch.arange(self.n_visual_tokens))
            input_ids = torch.cat([input_ids, visual_ids], dim=0)

        elif self.context == "text_only":
            page_info = self.get_page_info(d)
            section_info, labels = self.get_section_info(section_id, d, remove_summary=True)
            context_info = []
            for context_id in range(len(d['section_title'])):
                if context_id == section_id:
                    continue
                context_info.append(self.get_section_info(context_id, d, remove_summary=False))
            context_info = ', '.join(context_info)
            inputs = "summarize: " + section_info + ", context: " + page_info + context_info
            input_ids = self.tokenizer(inputs, max_length=self.max_input_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0]
        
        elif self.context == "all": #graph token will be added here
            page_info = self.get_page_info(d)
            section_info, labels = self.get_section_info(section_id, d, remove_summary=True)
            section_image, section_caption = self.get_section_images(page_id, section_id, d)
            images = []
            image_positions = []
             
            if section_image is None:
                inputs = "summarize: " + section_info
                visual_ids = torch.LongTensor(self.n_visual_tokens * [self.tokenizer.pad_token_id])
                images.append(torch.zeros((3,  224, 224)))
            else:
                inputs = "summarize: " + section_info + ", context: " + section_caption
                visual_ids = torch.LongTensor(self.n_visual_tokens * [100])
                images.append(section_image)
            
            max_text_length = self.max_input_length - self.n_visual_tokens - 1 #-1 for graph token
            input_ids = self.tokenizer(inputs, max_length=max_text_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0]
            # Add graph token at the beginning
            graph_token = torch.LongTensor([100])
            input_ids = torch.cat([graph_token, input_ids], dim=0)
            #add image
            image_positions.append(input_ids.shape[0] + torch.arange(self.n_visual_tokens))
            input_ids = torch.cat([input_ids, visual_ids], dim=0)

            for context_id in range(len(d['section_title'])):
                if context_id == section_id:
                    continue
                context_info = self.get_section_info(context_id, d, remove_summary=False)
                context_image, context_caption = self.get_section_images(page_id, context_id, d)
                if len(images) < self.max_image_neighbors:
                    max_text_length = self.max_input_length - input_ids.shape[0] - self.n_visual_tokens
                    if max_text_length <= 2:
                        break
                    if context_image is None:
                        context = context_info
                        visual_ids = torch.LongTensor(self.n_visual_tokens * [self.tokenizer.pad_token_id])
                        images.append(torch.zeros((3,  224, 224)))
                    else:
                        context = context_info + context_caption
                        visual_ids = torch.LongTensor(self.n_visual_tokens * [100])
                        images.append(context_image)
                    context_ids = self.tokenizer(context, max_length=max_text_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0][1:]
                    image_positions.append(input_ids.shape[0] + context_ids.shape[0] + torch.arange(self.n_visual_tokens))
                    input_ids = torch.cat([input_ids, context_ids, visual_ids], dim=0)
                else:
                    max_text_length = self.max_input_length - input_ids.shape[0]
                    if max_text_length <= 2:
                        break
                    context_ids = self.tokenizer(context_info, max_length=max_text_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0][1:]
                    input_ids = torch.cat([input_ids, context_ids], dim=0)

            while len(images) < self.max_image_neighbors:
                images.append(torch.zeros((3,  224, 224)))
                image_positions.append(-1 * torch.ones((self.n_visual_tokens)))

            if len(input_ids) > self.max_input_length:
                input_ids = input_ids[:self.max_input_length]

        if self.decoder_only:
            model_inputs = self.tokenizer.pad({"input_ids": [input_ids]}, max_length=self.max_input_length, padding="max_length", return_tensors="pt")
            labels = ", summary: " + labels
            label_ids = self.tokenizer(labels, max_length=self.max_output_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0]
            # Remove SOS token and add EOS token
            label_ids = torch.cat([label_ids[1:], torch.LongTensor([self.tokenizer.eos_token_id])], dim=0)
            model_outputs = self.tokenizer.pad({"input_ids": [label_ids]}, max_length=self.max_output_length, padding="max_length", return_tensors="pt")

            result = {"input_ids": torch.cat((model_inputs.input_ids[0], model_outputs.input_ids[0]), dim=0),\
                      "attention_mask": torch.cat((model_inputs.attention_mask[0], model_outputs.attention_mask[0]), dim=0),\
                      "labels": torch.cat((model_inputs.input_ids[0], model_outputs.input_ids[0]), dim=0)}
        else:
            model_inputs = self.tokenizer.pad({"input_ids": [input_ids]}, max_length=self.max_input_length, padding="max_length", return_tensors="pt")
            labels = self.tokenizer(labels, max_length=self.max_output_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
            labels_with_ignore_index = torch.LongTensor([label if label != 0 else -100 for label in labels])
            result = {"input_ids": model_inputs.input_ids[0], "attention_mask": model_inputs.attention_mask[0], "labels": labels_with_ignore_index}

        if self.context in ("section_all", "all"):
            images = torch.stack(images, dim=0)
            image_positions = torch.cat(image_positions, dim=0).long()
            result["images"] = images
            result["image_positions"] = image_positions

        if self.context == "all":
            # Multimodal neighbor information
            neighbor_texts = []
            neighbor_images = []
            position_texts = []
            position_images = []
            location_texts = []
            location_images = []
            location = 0
            # Graph
            graph_index = {section_id: 0} # input text: 0, neighbors: location + 1
            edge_list = []
            
            #(input node itself)
            neighbor_texts.append(section_info)
            position_texts.append(len(position_texts))
            location_texts.append(location)
            location += 1
            
            #(1) page information
            page_info = self.get_page_info(d)
            neighbor_texts.append(page_info)
            position_texts.append(len(position_texts))
            location_texts.append(location)
            location += 1
            # Graph: input_text <-> page description
            edge_list.append((graph_index[section_id], location))

            #(2) session image information
            if self.context != "text_only":
                section_image, section_caption = self.get_section_images(page_id, section_id, d)
                if section_image is not None:
                    neighbor_texts.append(section_caption)
                    position_texts.append(len(position_texts))
                    location_texts.append(location)
                    location += 1
                    # Graph: input_text <-> caption
                    edge_list.append((graph_index[section_id], location))
                    # Graph: image <-> caption
                    # edge_list.append((previous_image_id, location))

            #(3) rest section information
            if self.context != "section_all":
                previous_section_id = 1 # page
                for context_id in range(len(d['section_title'])):
                    if context_id == section_id:
                        continue
                    if len(neighbor_texts) < self.max_text_neighbors + 1:
                        context_info = self.get_section_info(context_id, d, remove_summary=False)
                        neighbor_texts.append(context_info)
                        position_texts.append(len(position_texts))
                        location_texts.append(location)
                        location += 1
                        # Graph: previous section - current section (order)
                        edge_list.append((previous_section_id, location))
                        graph_index[context_id] = location
                        previous_section_id = location

                    if self.context != "text_only":
                        if len(neighbor_images) < self.max_image_neighbors:
                            context_image, context_caption = self.get_section_images(page_id, context_id, d)
                            if context_image is not None:
                                if len(neighbor_texts) < self.max_text_neighbors + 1:
                                    neighbor_texts.append(context_caption)
                                    position_texts.append(len(position_texts))
                                    location_texts.append(location)
                                    location += 1
                                    # Graph: section <-> caption
                                    edge_list.append((previous_section_id, location))

            # Graph: hierachical relations
            for context_id in range(len(d['section_parent_index'])):
                parent_id = d['section_parent_index'][context_id]
                if context_id in graph_index.keys() and parent_id in graph_index.keys():
                    edge_list.append((graph_index[context_id], graph_index[parent_id]))

            # PyG graph data
            # node_num = 1 + self.max_text_neighbors + self.max_image_neighbors
            node_num = 1 + self.max_text_neighbors
            edge_index = torch.LongTensor(edge_list).t().contiguous()
            if self.position_type == 'laplacian':
                node_value = torch.zeros((node_num))
                graph = Data(x=node_value, edge_index=edge_index)
                lpe = utils.compute_LPE(graph)
                
            edge_value = torch.ones((edge_index.shape[1]))
            graph = torch.sparse_coo_tensor(edge_index, edge_value, [node_num, node_num]).to_dense()
            graph = utils.normalize_graph(graph)
            
            graph = (graph > 0).to(torch.long)
            
            # Increase position ids by 1 for padding_id
            position_texts = [position_id + 1 for position_id in position_texts]
            # Pad
            while len(neighbor_texts) < self.max_text_neighbors + 1:
                neighbor_texts.append('')
                position_texts.append(0)
                location_texts.append(location)
                location += 1

            #Tokenize
            # neighbor_max_length = 77 if "clip" in self.text_model else 512
            neighbor_max_length = 512
            neighbor_texts = self.tokenizer(neighbor_texts, max_length=neighbor_max_length, padding="max_length", truncation=True, return_tensors="pt")
            result["neighbor_input_ids"] = neighbor_texts.input_ids,
            result["neighbor_attention_mask"] = neighbor_texts.attention_mask,
            result["graph"] = graph
        # print(result.keys())
        return result

    def get_embedding_item(self, index):
        page_id, section_id = self.id_list[index]
        d = self.df[self.df['page_id'] == page_id].iloc[0]

        section_info, labels = self.get_section_info(section_id, d, remove_summary=True)
        inputs = "summarize: " + section_info
        if self.context != "text_only":
            total_neighbor_tokens = self.max_text_neighbors * self.n_text_tokens + self.max_image_neighbors * self.n_visual_tokens
        else:
            total_neighbor_tokens = self.max_text_neighbors * self.n_text_tokens
        max_text_length = self.max_input_length - total_neighbor_tokens
        input_ids = self.tokenizer(inputs, max_length=max_text_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0]
        model_inputs = self.tokenizer.pad({"input_ids": [input_ids]}, max_length=self.max_input_length, padding="max_length", return_tensors="pt")

        if self.decoder_only:
            labels = ", summary: " + labels
            label_ids = self.tokenizer(labels, max_length=self.max_output_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0]
            # Remove SOS token and add EOS token
            label_ids = torch.cat([label_ids[1:], torch.LongTensor([self.tokenizer.eos_token_id])], dim=0)
            model_outputs = self.tokenizer.pad({"input_ids": [label_ids]}, max_length=self.max_output_length, padding="max_length", return_tensors="pt")

            result = {"input_ids": torch.cat((model_inputs.input_ids[0], model_outputs.input_ids[0]), dim=0), \
                    "attention_mask": torch.cat((model_inputs.attention_mask[0], model_outputs.attention_mask[0]), dim=0), \
                    "labels": torch.cat((model_inputs.input_ids[0], model_outputs.input_ids[0]), dim=0)}
        else:
            labels = self.tokenizer(labels, max_length=self.max_output_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
            labels_with_ignore_index = torch.LongTensor([label if label != 0 else -100 for label in labels])
            result = {"input_ids": model_inputs.input_ids[0], "attention_mask": model_inputs.attention_mask[0], "labels": labels_with_ignore_index}

        # Multimodal neighbor information
        neighbor_texts = []
        neighbor_images = []
        position_texts = []
        position_images = []
        location_texts = []
        location_images = []
        location = 0
        # Graph
        graph_index = {section_id: 0} # input text: 0, neighbors: location + 1
        edge_list = []

        #(1) page information
        page_info = self.get_page_info(d)
        neighbor_texts.append(page_info)
        position_texts.append(len(position_texts))
        location_texts.append(location)
        location += 1
        # Graph: input_text <-> page description
        edge_list.append((graph_index[section_id], location))

        #(2) session image information
        if self.context != "text_only":
            section_image, section_caption = self.get_section_images(page_id, section_id, d)
            if section_image is not None:
                neighbor_images.append(section_image)
                position_images.append(len(position_images))
                location_images.append(location)
                location += 1
                # Graph: input_text <-> image
                edge_list.append((graph_index[section_id], location))
                previous_image_id = location

                neighbor_texts.append(section_caption)
                position_texts.append(len(position_texts))
                location_texts.append(location)
                location += 1
                # Graph: input_text <-> caption
                edge_list.append((graph_index[section_id], location))
                # Graph: image <-> caption
                edge_list.append((previous_image_id, location))

        #(3) rest section information
        if self.context != "section_all":
            previous_section_id = 1 # page
            for context_id in range(len(d['section_title'])):
                if context_id == section_id:
                    continue
                if len(neighbor_texts) < self.max_text_neighbors:
                    context_info = self.get_section_info(context_id, d, remove_summary=False)
                    neighbor_texts.append(context_info)
                    position_texts.append(len(position_texts))
                    location_texts.append(location)
                    location += 1
                    # Graph: previous section - current section (order)
                    edge_list.append((previous_section_id, location))
                    graph_index[context_id] = location
                    previous_section_id = location

                if self.context != "text_only":
                    if len(neighbor_images) < self.max_image_neighbors:
                        context_image, context_caption = self.get_section_images(page_id, context_id, d)
                        if context_image is not None:
                            neighbor_images.append(context_image)
                            position_images.append(len(position_images))
                            location_images.append(location)
                            location += 1
                            # Graph: section <-> image
                            edge_list.append((previous_section_id, location))
                            previous_image_id = location

                            if len(neighbor_texts) < self.max_text_neighbors:
                                neighbor_texts.append(context_caption)
                                position_texts.append(len(position_texts))
                                location_texts.append(location)
                                location += 1
                                # Graph: section <-> caption
                                edge_list.append((previous_section_id, location))
                                # Graph: image <-> caption
                                edge_list.append((previous_image_id, location))

        # Graph: hierachical relations
        for context_id in range(len(d['section_parent_index'])):
            parent_id = d['section_parent_index'][context_id]
            if context_id in graph_index.keys() and parent_id in graph_index.keys():
                edge_list.append((graph_index[context_id], graph_index[parent_id]))

        # PyG graph data
        node_num = 1 + self.max_text_neighbors + self.max_image_neighbors
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        if self.position_type == 'laplacian':
            node_value = torch.zeros((node_num))
            graph = Data(x=node_value, edge_index=edge_index)
            lpe = utils.compute_LPE(graph)
        elif self.position_type == 'gnn':
            edge_value = torch.ones((edge_index.shape[1]))
            graph = torch.sparse_coo_tensor(edge_index, edge_value, [node_num, node_num]).to_dense()
            graph = utils.normalize_graph(graph)

        # Increase position ids by 1 for padding_id
        position_texts = [position_id + 1 for position_id in position_texts]
        # Pad
        while len(neighbor_texts) < self.max_text_neighbors:
            neighbor_texts.append('')
            position_texts.append(0)
            location_texts.append(location)
            location += 1

        if self.context != "text_only":
            position_images = [position_id + 1 for position_id in position_images]
            while len(neighbor_images) < self.max_image_neighbors:
                neighbor_images.append(torch.zeros((3,  224, 224)))
                position_images.append(0)
                location_images.append(location)
                location += 1

        #Tokenize
        neighbor_max_length = 77 if "clip" in self.text_model else 512
        neighbor_texts = self.text_tokenizer(neighbor_texts, max_length=neighbor_max_length, padding="max_length", truncation=True, return_tensors="pt")
        result["neighbor_input_ids"] = neighbor_texts.input_ids,
        result["neighbor_attention_mask"] = neighbor_texts.attention_mask,
        result["neighbor_pos_ids"] = torch.LongTensor(position_texts),
        result["text_locations"] = torch.LongTensor(location_texts),
        if self.context != "text_only":
            result["neighbor_images"] = torch.stack(neighbor_images, dim=0),
            result["neighbor_images_pos_ids"] = torch.LongTensor(position_images)
            result["image_locations"] = torch.LongTensor(location_images),
        if self.position_type == 'laplacian':
            result["lpe"] = lpe
        if self.position_type == 'gnn':
            result["graph"] = graph
        return result

def collate(items):
    input_ids = []
    attention_mask = []
    labels = []
    images = []
    image_positions = []
    for item in items:
        input_ids.append(item["input_ids"])
        attention_mask.append(item["attention_mask"])
        labels.append(item["labels"])
        images.append(item["images"])
        image_positions.append(item["image_positions"])
    return {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "images": torch.stack(images, dim=0),
            "labels": torch.stack(labels, dim=0),
            "image_positions": torch.stack(image_positions, dim=0),
            }


