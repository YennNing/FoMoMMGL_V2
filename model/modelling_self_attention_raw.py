import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    CLIPVisionModel,
    CLIPTextModel
)

from peft import (
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningInit,
    PromptTuningConfig,
    TaskType,
    get_peft_model,
)

from .graph import GCN

class TextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SelfAttentionModel(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()

        self.args = args
        self.context = args.context
        self.decoder_only = args.decoder_only
        self.neighbor_mode = args.neighbor_mode
        self.position_type = args.position_type
        self.n_text_tokens = args.n_text_tokens
        self.n_visual_tokens = args.n_visual_tokens
        self.n_virtual_tokens = args.n_virtual_tokens
        self.tokenizer = tokenizer

        if "t5" in args.model_name_or_path:
            peft_task_type = TaskType.SEQ_2_SEQ_LM
            config = AutoConfig.from_pretrained(args.model_name_or_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, config=config)
        elif "opt" in args.model_name_or_path:
            peft_task_type = TaskType.CAUSAL_LM
            config = AutoConfig.from_pretrained(args.model_name_or_path)
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config)
        else:
            raise ValueError(f"SelfAttentionModel does not support {args.model_name_or_path}.")

        if args.peft_type == "none":
            self.lm = model
        else:
            if args.peft_type == "lora":
                peft_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    modules_to_save=["lm_head"],
                )
            elif args.peft_type == "prefix":
                peft_config = PrefixTuningConfig(
                    task_type=peft_task_type,
                    inference_mode=False,
                    prefix_projection=True,
                    num_virtual_tokens=self.n_virtual_tokens
                )
            elif args.peft_type == "prompt":
                peft_config = PromptTuningConfig(
                    task_type=peft_task_type,
                    prompt_tuning_init=PromptTuningInit.RANDOM,
                    num_virtual_tokens=self.n_virtual_tokens
                )
            else:
                raise ValueError(f"SelfAttentionModel does not support {args.peft_type}.")
            self.lm = get_peft_model(model, peft_config)

        self.input_embeddings = self.lm.get_input_embeddings()

        self.text_model = None
        if self.neighbor_mode == "prefix":
            # Text model processing text neighbors
            embedding_dim = self.input_embeddings.embedding_dim * args.n_text_tokens
            self.text_model = CLIPTextModel.from_pretrained(args.text_model)
            self.text_embeddings = nn.Linear(self.text_model.config.hidden_size, embedding_dim)
            if self.position_type == "none":
                self.text_position_embeddings = nn.Embedding(args.max_output_length + 1, embedding_dim) # + 1 for padding neighbors

            self.text_model.eval()
            for name, param in self.text_model.named_parameters():
                param.requires_grad = False

        self.visual_model = None
        if self.context in ("section_all", "all"):
            # Vision model processing image neighbors
            embedding_dim = self.input_embeddings.embedding_dim * args.n_visual_tokens
            self.visual_model = CLIPVisionModel.from_pretrained(args.visual_model)
            self.visual_embeddings = nn.Linear(self.visual_model.config.hidden_size, embedding_dim)
            if self.position_type == "none":
                self.visual_position_embeddings = nn.Embedding(args.max_output_length + 1, embedding_dim) # + 1 for padding neighbors

            self.visual_model.eval()
            for param in self.visual_model.parameters():
                param.requires_grad = False

        if self.position_type == "laplacian":
            if self.context in ("section_only", "section_all", "text_only") or self.neighbor_mode == "raw":
                raise ValueError(f"[Laplacian PE] neighbor mode: {self.neighbor_mode} and context: {self.context} are not supported.")
            k = 1 + args.max_text_neighbors + args.max_image_neighbors - 5
            embedding_dim = self.input_embeddings.embedding_dim * args.n_text_tokens
            self.lpe_embeddings = nn.Linear(k, embedding_dim)

        if self.neighbor_mode != 'raw' and self.position_type == "gnn":
            embedding_dim = self.input_embeddings.embedding_dim * args.n_text_tokens
            self.gnn = GCN(input_dim=embedding_dim, output_dim=embedding_dim, hidden_dim=self.text_model.config.hidden_size)

        # Freeze the base LM
        if self.args.freeze_lm:
            print("Freezing the LM.")
            self.lm.eval()
            for param in self.lm.parameters():
                param.requires_grad = False
        else:
            self.lm.train()

    def get_text_embs(self, input_ids, attention_mask, pos_ids):
        batch_size, neighbor_num, seq_len = input_ids.shape
        input_ids = input_ids.reshape(-1, seq_len)
        attention_mask = attention_mask.reshape(-1, seq_len)

        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        encoder_outputs = outputs.pooler_output
        text_embs = self.text_embeddings(encoder_outputs)

        if self.position_type == "none":
            pos_ids = pos_ids.reshape(-1)
            text_embs = text_embs + self.text_position_embeddings(pos_ids)

        text_embs = text_embs.reshape(text_embs.shape[0], self.n_text_tokens, -1)
        return text_embs.reshape(batch_size, neighbor_num, self.n_text_tokens, -1)

    def get_visual_embs(self, pixel_values, pos_ids=None):
        batch_size, neighbor_num, pixel, width, height = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, pixel, width, height)
        outputs = self.visual_model(pixel_values)
        encoder_outputs = outputs.pooler_output
        visual_embs = self.visual_embeddings(encoder_outputs)

        if self.position_type == "none" and pos_ids is not None:
            pos_ids = pos_ids.reshape(-1)
            visual_embs = visual_embs + self.visual_position_embeddings(pos_ids)

        visual_embs = visual_embs.reshape(visual_embs.shape[0], self.n_visual_tokens, -1)
        return visual_embs.reshape(batch_size, neighbor_num, self.n_visual_tokens, -1)

    def train(self, mode=True):
        super(SelfAttentionModel, self).train(mode=mode)
        if self.args.freeze_lm:
            self.lm.eval()
        if self.text_model is not None:
            self.text_model.eval()
        if self.visual_model is not None:
            self.visual_model.eval()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        images=None,
        image_positions=None,
        neighbor_input_ids=None,
        neighbor_attention_mask=None,
        neighbor_pos_ids=None,
        text_locations=None,
        neighbor_images=None,
        neighbor_images_pos_ids=None,
        image_locations=None,
        lpe=None,
        graph=None
    ):

        if self.neighbor_mode == "raw" and self.context in ("section_only", "text_only"):
            return self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        elif self.neighbor_mode == "raw" and self.context in ("section_all", "all"):
            input_embs = self.input_embeddings(input_ids)
            visual_embs = self.get_visual_embs(images)

            batch_size, seq_len, hidden_dim = input_embs.shape
            if self.context == "section_all":
                batch_idx = torch.arange(batch_size)[:, None]
                input_embs[batch_idx, image_positions] = visual_embs.reshape(batch_size, -1, hidden_dim)
                if self.decoder_only:
                    labels[batch_idx, image_positions] = -100
            else:
                for batch_idx in range(batch_size):
                    for image_idx in range(images.shape[1]):
                        image_position = image_positions[batch_idx][self.n_visual_tokens * image_idx: self.n_visual_tokens * (image_idx + 1)]
                        if image_position.sum() == -1 * self.n_visual_tokens:
                            continue
                        input_embs[batch_idx, image_position] = visual_embs[batch_idx, image_idx]
                        if self.decoder_only:
                            labels[batch_idx, image_position] = -100

            return self.lm(inputs_embeds=input_embs, attention_mask=attention_mask, labels=labels)

        elif self.neighbor_mode == "prefix" and self.context in ("section_only", "text_only"):
            batch_size, neighbor_num, seq_len = neighbor_input_ids.shape
            neighbor_embeds = self.get_text_embs(neighbor_input_ids, neighbor_attention_mask, neighbor_pos_ids)
            neighbor_embeds = neighbor_embeds.reshape(batch_size, neighbor_num * self.n_text_tokens, -1)
            neighbor_attention_mask = neighbor_pos_ids > 0
            neighbor_attention_mask = torch.repeat_interleave(neighbor_attention_mask, repeats=self.n_text_tokens, dim=1)

            neighbor_start = self.args.max_input_length - neighbor_num * self.n_text_tokens
            neighbor_end = self.args.max_input_length
            input_embs = self.input_embeddings(input_ids)
            input_embs[:, neighbor_start:neighbor_end] = neighbor_embeds
            attention_mask[:, neighbor_start:neighbor_end] = neighbor_attention_mask

            if self.decoder_only:
                labels[:, neighbor_start:neighbor_end] = -100

            return self.lm(inputs_embeds=input_embs, attention_mask=attention_mask, labels=labels)

        elif self.neighbor_mode == "prefix" and self.context in ("section_all", "all"):
            text_embeds = self.get_text_embs(neighbor_input_ids, neighbor_attention_mask, neighbor_pos_ids)
            batch_size, text_neighbor_num, n_tokens, hidden_dim = text_embeds.shape
            text_attention_mask = neighbor_pos_ids > 0
            text_attention_mask = text_attention_mask.unsqueeze(-1).expand(-1, -1, self.n_text_tokens)

            visual_embeds = self.get_visual_embs(neighbor_images, neighbor_images_pos_ids)
            batch_size, visual_neighbor_num, n_tokens, hidden_dim = visual_embeds.shape
            visual_attention_mask = neighbor_images_pos_ids > 0
            visual_attention_mask = visual_attention_mask.unsqueeze(-1).expand(-1, -1, self.n_visual_tokens)

            batch_idx = torch.arange(batch_size)[:, None]
            total_neighbor_num = text_neighbor_num + visual_neighbor_num
            neighbor_embeds = torch.zeros((batch_size, total_neighbor_num, n_tokens, hidden_dim)).to(neighbor_input_ids.device)
            neighbor_embeds[batch_idx, text_locations] = text_embeds
            neighbor_embeds[batch_idx, image_locations] = visual_embeds
            neighbor_embeds = neighbor_embeds.reshape(batch_size, -1, hidden_dim)

            neighbor_attention_mask = torch.zeros((batch_size, total_neighbor_num, n_tokens)).bool().to(neighbor_attention_mask.device)
            neighbor_attention_mask[batch_idx, text_locations] = text_attention_mask
            neighbor_attention_mask[batch_idx, image_locations] = visual_attention_mask
            neighbor_attention_mask = neighbor_attention_mask.reshape(batch_size, -1)

            neighbor_start = self.args.max_input_length - total_neighbor_num * n_tokens
            neighbor_end = self.args.max_input_length
            input_embs = self.input_embeddings(input_ids)
            if self.context == "all":
                if self.position_type == "laplacian":
                    lpe_embeddings = self.lpe_embeddings(lpe)
                    lpe_embeddings = lpe_embeddings.reshape(batch_size, total_neighbor_num + 1, n_tokens, hidden_dim)
                    neighbor_embeds = neighbor_embeds + lpe_embeddings[:, 1:].reshape(batch_size, -1, hidden_dim)
                elif self.position_type == "gnn":
                    neighbor_embeds = neighbor_embeds.view(batch_size, total_neighbor_num, n_tokens, hidden_dim).view(batch_size, total_neighbor_num, -1)
                    gnn_embeds = self.gnn(neighbor_embeds, graph)
                    neighbor_embeds = neighbor_embeds + gnn_embeds
                    neighbor_embeds = neighbor_embeds.view(batch_size, total_neighbor_num, n_tokens, hidden_dim).view(batch_size, -1, hidden_dim)
                    
                    
            input_embs[:, neighbor_start:neighbor_end] = neighbor_embeds
            attention_mask[:, neighbor_start:neighbor_end] = neighbor_attention_mask

            if self.decoder_only:
                labels[:, neighbor_start:neighbor_end] = -100
            return self.lm(inputs_embeds=input_embs, attention_mask=attention_mask, labels=labels)

        else:
            raise ValueError(f"Neighbor mode: {self.neighbor_mode} and context: {self.context} are not supported.")

