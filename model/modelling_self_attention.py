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
        hidden_dim = self.input_embeddings.embedding_dim
        num_heads = 8
        num_layers = 12
        dropout = 0.1

        self.text_model = None
        if self.neighbor_mode == "prefix":
            # Text model processing text neighbors
            embedding_dim = self.input_embeddings.embedding_dim * args.n_text_tokens
            # self.text_model = CLIPTextModel.from_pretrained(args.text_model)
            self.text_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config)
            self.text_embeddings = nn.Linear(self.text_model.config.hidden_size, embedding_dim)
            if self.position_type == "none":
                self.text_position_embeddings = nn.Embedding(args.max_output_length + 1, embedding_dim) # + 1 for padding neighbors

            self.text_model.eval()
            for name, param in self.text_model.named_parameters():
                param.requires_grad = False
        
        
        self.visual_model = None
        if self.context in ("section_all", "all"):
            embedding_dim = self.input_embeddings.embedding_dim
            # self.neighbor_text_model = CLIPTextModel.from_pretrained(args.text_model)
            self.text_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config)
            self.text_embeddings = nn.Linear(self.text_model.config.hidden_size, embedding_dim)
            if self.position_type == "none":
                self.text_position_embeddings = nn.Embedding(args.max_output_length + 1, embedding_dim) # + 1 for padding neighbors
            
            if self.context == "all":
                self.gnn = GCN(hidden_dim=embedding_dim)

            
            # self.neighbor_text_model.eval()
            # for name, param in self.neighbor_text_model.named_parameters():
            #     param.requires_grad = False
            
            self.text_model.eval()
            for name, param in self.text_model.named_parameters():
                param.requires_grad = False
                
            #  # Transformer Layers: Self-Attention, Cross-Attention, Feed-Forward
            # self.layers = nn.ModuleList([
            #     nn.ModuleDict({
            #         "self_attn": nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True),
            #         "cross_attn": nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True),
            #         "feed_forward": nn.Sequential(
            #             nn.Linear(hidden_dim, hidden_dim * 4),
            #             nn.GELU(),
            #             nn.Linear(hidden_dim * 4, hidden_dim),
            #             nn.Dropout(dropout),
            #         ),
            #         "norm1": nn.LayerNorm(hidden_dim),
            #         "norm2": nn.LayerNorm(hidden_dim),
            #         "norm3": nn.LayerNorm(hidden_dim),
            #     })
            #     for _ in range(num_layers)
            # ])

            # Vision model processing image neighbors
            self.n_heads = 8
            self.n_layers = 12
            self.mhasat = nn.Transformer(
                            d_model= self.text_model.config.hidden_size,
                            nhead=self.n_heads,
                            num_encoder_layers=self.n_layers,
                            num_decoder_layers=0, #self.n_layers
                            dim_feedforward=self.text_model.config.hidden_size * 4, #256 * 4
                            dropout=0.1,
                        )
            self.text_cross_attention = nn.MultiheadAttention(embed_dim= self.text_model.config.hidden_size, num_heads= self.n_heads)
            self.visual_cross_attention = nn.MultiheadAttention(embed_dim= self.text_model.config.hidden_size, num_heads= self.n_heads)
                
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

    # def get_text_embs(self, input_ids, attention_mask, pos_ids):
    #     batch_size, neighbor_num, seq_len = input_ids.shape
    #     input_ids = input_ids.reshape(-1, seq_len)
    #     attention_mask = attention_mask.reshape(-1, seq_len)

    #     outputs = self.neighbor_text_model(input_ids=input_ids, attention_mask=attention_mask)
    #     encoder_outputs = outputs.pooler_output #([24, 512])
    #     # print("encoder outputs:",encoder_outputs.shape)
    #     text_embs = self.text_embeddings(encoder_outputs)
        
    #     # print('text_embs shape:', text_embs.shape) #torch.Size([24, 3072])
        
    #     return text_embs.reshape(batch_size, neighbor_num, -1)
    
    
    def get_text_embs(self, input_ids, attention_mask, pos_ids): #using the original text model for getting the embedding
        batch_size, neighbor_num, seq_len = input_ids.shape
        input_ids = input_ids.reshape(-1, seq_len)
        attention_mask = attention_mask.reshape(-1, seq_len)
        text_outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        text_embs = text_outputs.hidden_states[-1]  # (B, seq_len, D)
        text_embs = text_embs.mean(dim=1)  # (B, D)
        return text_embs.reshape(batch_size, neighbor_num, -1)


    def get_visual_embs(self, input_ids, attention_mask, pixel_values):
        # Self Attention on input_ids
        batch_size, seq_len = input_ids.shape
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        text_encoder_outputs = text_outputs.hidden_states[-1]  # (B, seq_len, D)
        text_embs = text_encoder_outputs.mean(dim=1)  # (B, D)

        # Process visual neighbors
        batch_size, visual_neighbor_num, pixel, width, height = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, pixel, width, height)  # (B * visual_neighbor_num, pixel, H, W)
        visual_outputs = self.visual_model(pixel_values)
        visual_encoder_outputs = visual_outputs.pooler_output  # (B * visual_neighbor_num, D)
        visual_embs = visual_encoder_outputs.reshape(batch_size, visual_neighbor_num, -1)  # (B, visual_neighbor_num, D)

        # Replicate text_embs to match visual_neighbor_num
        text_embs = text_embs.unsqueeze(1).repeat(1, visual_neighbor_num, 1)  # (B, visual_neighbor_num, D)
        text_embs = text_embs.transpose(0, 1)  # (visual_neighbor_num, B, D)

        # Cross Attention: input_ids (text) as query, visual_embs as key/value
        visual_embs = visual_embs.transpose(0, 1)  # (visual_neighbor_num, B, D)
        H_final = self.visual_cross_attention(
            query=text_embs,  # (visual_neighbor_num, B, D)
            key=visual_embs,  # (visual_neighbor_num, B, D)
            value=visual_embs  # (visual_neighbor_num, B, D)
        )[0]  # (visual_neighbor_num, B, D)

        # Transpose back and reshape
        H_final = H_final.transpose(0, 1)  # (B, visual_neighbor_num, D)
        visual_embs = self.visual_embeddings(H_final)  # (B, visual_neighbor_num, embedding_dim)

        # Return reshaped embeddings
        return visual_embs.reshape(batch_size, visual_neighbor_num, self.n_visual_tokens, -1)
        # Process text input
        # batch_size, seq_len = input_ids.shape
        # text_outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # text_embs = text_outputs.hidden_states[-1]  # (B, seq_len, D)
        # text_embs = text_embs.mean(dim=1)  # (B, D)

        # batch_size, visual_neighbor_num, pixel, width, height = pixel_values.shape
        # pixel_values = pixel_values.reshape(-1, pixel, width, height)  # (B * visual_neighbor_num, pixel, H, W)
        # visual_outputs = self.visual_model(pixel_values)
        # visual_embs = visual_outputs.pooler_output  # (B * visual_neighbor_num, D)
        # visual_embs = visual_embs.reshape(batch_size, visual_neighbor_num, -1)  # (B, visual_neighbor_num, D)

        # text_embs = text_embs.unsqueeze(1).repeat(1, visual_neighbor_num, 1)  # (B, visual_neighbor_num, D)

        # # print(text_embs.shape)
        # # print(visual_neighbor_num)
        # for layer in self.layers:
        #     text_embs = layer["self_attn"](query=text_embs, key=text_embs, value=text_embs)[0]
        #     text_embs = layer["norm1"](text_embs)

        #     cross_embs = layer["cross_attn"](query=text_embs, key=visual_embs, value=visual_embs)[0]
        #     cross_embs = layer["norm2"](cross_embs)

        #     cross_embs = layer["feed_forward"](cross_embs)
        #     cross_embs = layer["norm3"](cross_embs)

        # visual_embs = self.visual_embeddings(cross_embs)  # (B, visual_neighbor_num, D)
        # return visual_embs.reshape(batch_size, visual_neighbor_num, self.n_visual_tokens, -1)
        


    def train(self, mode=True):
        super(SelfAttentionModel, self).train(mode=mode)
        if self.args.freeze_lm:
            self.lm.eval()
        if self.text_model is not None:
            self.text_model.eval()
            # self.neighbor_text_model.eval()
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
            visual_embs = self.get_visual_embs(input_ids, attention_mask, images) #input_ids, attention_mask, pixel_values, neighbor_input_ids, neighbor_attention_mask, pos_ids=None
            
            batch_size, seq_len, hidden_dim = input_embs.shape
            if self.context == "section_all":
                batch_idx = torch.arange(batch_size)[:, None]
                input_embs[batch_idx, image_positions] = visual_embs.reshape(batch_size, -1, hidden_dim)
                if self.decoder_only:
                    labels[batch_idx, image_positions] = -100
            else:
                # print('neighbor_input_ids')
                # print(neighbor_input_ids)
                # print('neighbor_attention_mask')
                # print(neighbor_attention_mask)
                batch_size, neighbor_num, seq_len = neighbor_input_ids.shape
                text_embeds = self.get_text_embs(neighbor_input_ids, neighbor_attention_mask, neighbor_pos_ids) 
                # print('gnn input', text_embeds.shape, text_embeds) # torch.Size([2, 12,768])
                
                gnn_embeds = self.gnn(text_embeds, graph) 
                row_mask = graph.sum(dim=-1) > 0  # Shape: (batch_size, num_nodes)
                filtered_gnn_embeds = [gnn_embeds[i][row_mask[i]] for i in range(gnn_embeds.size(0))]  # List of tensors
                graph_token = torch.stack([
                    gnn_sample.mean(dim=0) if gnn_sample.size(0) > 0 else torch.zeros(gnn_embeds.size(-1), device=gnn_embeds.device)
                    for gnn_sample in filtered_gnn_embeds
                ])  # Shape: (batch_size, hidden_dim)
                graph_token = gnn_embeds.mean(dim=1)   # (batch_size, hidden_dim)
                input_embs[:, 0, :] = graph_token
                
                # print('graph_token', graph_token)
                # print(input_embs)
                
                for batch_idx in range(batch_size):
                    for image_idx in range(images.shape[1]):
                        image_position = image_positions[batch_idx][self.n_visual_tokens * image_idx: self.n_visual_tokens * (image_idx + 1)]
                        if image_position.sum() == -1 * self.n_visual_tokens:
                            continue
                        input_embs[batch_idx, image_position] = visual_embs[batch_idx, image_idx]
                        if self.decoder_only:
                            labels[batch_idx, image_position] = -100
                            labels[batch_idx, 0] = -100  #for graph token
              
            # print(input_embs.shape)
            # print(attention_mask.shape)
            # print(labels.shape)
                        
            return self.lm(inputs_embeds=input_embs, attention_mask=attention_mask, labels=labels)

        elif self.neighbor_mode == "prefix" and self.context in ("section_only", "text_only"):
            batch_size, neighbor_num, seq_len = neighbor_input_ids.shape
            neighbor_embeds = self.get_text_embs(neighbor_input_ids, neighbor_attention_mask, neighbor_pos_ids)  #(BS, neighbor_num, token_num, embedding)
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

