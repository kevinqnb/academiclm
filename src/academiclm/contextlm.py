import os
import gc
from tqdm import tqdm
import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from nnsight import LanguageModel
from nnterp import StandardizedTransformer
from .utils import tokenize, jensen_shannon_divergence
from scipy.spatial.distance import jensenshannon


class ContextLM:
    """
    A wrapper around NNsight language models that provides methods for generating text
    and computing hallucination scores based upon input context and instructions.

    This is intended to be an application of methods described in the following paper:
    Sun, Zhongxiang, et al. "ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation
    via Mechanistic Interpretability." ICLR. 2025.

    Args:
        model_name (str): The name of the model to load from NNsight or huggingface.
        top_k (float): The fraction of context tokens with largest attention weight to 
            compare generated tokens with (for external context score). Default is 0.1 (10%).
        sampling_params (dict): A dictionary of sampling parameters to pass to the
            NNsight LanguageModel generate method. Default is {}.
        nnsight_kwargs (dict): Additional keyword arguments to pass to the NNsight LanguageModel.
        return_full_output (bool): Whether to return full per-layer and per-head scores
            in the output dictionary. Default is False.
        verbose (bool): Whether to print verbose output during generation. Default is False.
    """
    def __init__(
        self,
        model_name : str,
        top_k : int = 10,
        sampling_params : dict = {},
        nnsight_kwargs : dict = {},
        verbose : bool = False,
        generate_full_output : bool = False,
        cache_dir : str = None
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.sampling_params = {'max_new_tokens': 50} | sampling_params
        self.max_new_tokens = self.sampling_params['max_new_tokens']
        self.verbose = verbose
        self.generate_full_output = generate_full_output
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Detect available GPUs and set up device allocation
        self._setup_devices()
        
        self.llm = StandardizedTransformer(model_name, enable_attention_probs=True, **nnsight_kwargs) #device_map={"": "cuda:1"})
        self.tokenizer = self.llm.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.n_layers = len(self.llm.model.layers)
        self.n_heads = self.llm.config.num_attention_heads
        self.n_kv_heads = self.llm.config.num_key_value_heads
        self.head_dim = self.llm.config.hidden_size // self.n_heads

        self.responses = []
        self.parametric_score_arrays = []
        self.context_score_array = []


    def set_output_mode(
        self,
        generate_full_output : bool
    ):
        """
        Set whether to return full per-layer and per-head scores in the output dictionary.

        Args:
            generate_full_output (bool): Whether to return full per-layer and per-head scores
                in the output dictionary.
        """
        self.generate_full_output = generate_full_output

    def _setup_devices(self):
        """
        Set up device allocation for LLM and tensors.
        If multiple GPUs are available, use separate devices for LLM and tensors.
        Otherwise, use the same device for both.
        """
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if (num_gpus >= 2):
                # Use different GPUs for LLM and tensors
                self.llm_device = torch.device("cuda:0")
                self.tensor_device = torch.device("cuda:1")
                if self.verbose:
                    print(f"Using {num_gpus} GPUs: LLM on cuda:0, tensors on cuda:1")
            else:
                # Single GPU: use same device for both
                self.llm_device = torch.device("cuda:0")
                self.tensor_device = torch.device("cuda:0")
                if self.verbose:
                    print(f"Using single GPU: cuda:0")
        else:
            # CPU fallback
            self.llm_device = torch.device("cpu")
            self.tensor_device = torch.device("cpu")
            if self.verbose:
                print("No GPU available, using CPU")
        


    def compute_external_context_score(
        self,
        response_embeddings : torch.Tensor,
        context_embeddings : torch.Tensor,
        context_indices : torch.Tensor,
        attention_probabilities : list[torch.Tensor],
        prompt_len : int
    ) -> torch.Tensor:
        """
        Compute the external context score as the cosine similarity between the last token embedding
        and the mean of the top-k context embeddings.

        Args:
            last_token_emb (torch.Tensor): The embeddings of the generated response.
            context_emb_cache (torch.Tensor): The cached embeddings of the context tokens.
            attention_probabilities (list[torch.Tensor]): A list of attention probability tensors
                from each layer of the model.

        Returns:
            torch.Tensor: A tensor of shape [num_layers, num_heads] containing the external context scores.
        """
        k = min(self.top_k, len(context_embeddings))
        # Create scores tensor on same device as inputs for bfloat16 computation
        device = response_embeddings.device
        external_context_scores = torch.zeros(
            (len(response_embeddings), self.n_layers, self.n_heads), 
            device=device, 
            dtype=torch.bfloat16
        )

        for token_idx in range(len(response_embeddings)):
            for layer_idx in range(self.n_layers):
                A = attention_probabilities[token_idx, layer_idx, :, :prompt_len + token_idx]
                for head_idx in range(self.n_heads):
                    attn_weights = A[head_idx, context_indices]  # Shape: [seq_len]
                    top_k_indices = torch.topk(attn_weights, k).indices
                    top_k_emb = context_embeddings[top_k_indices]  # Shape: [k, hidden_size]
                    mean_top_k_emb = torch.mean(top_k_emb, dim=0)  # Shape: [hidden_size]
                    cosine_similarity = F.cosine_similarity(
                        mean_top_k_emb,
                        response_embeddings[token_idx],
                        dim=-1
                    )
                    external_context_scores[token_idx, layer_idx, head_idx] = cosine_similarity

        return external_context_scores.float().cpu().numpy()


    def compute_parametric_knowledge_score(
        self,
        mlp_inputs : torch.Tensor,
        mlp_outputs : torch.Tensor
    ) -> float:
        """
        Compute the parametric knowledge score as the Jensen-Shannon Divergence between
        the MLP input and output distributions.

        Args:
            mlp_input (torch.Tensor): The input to the MLP layer (before transformation).
            mlp_output (torch.Tensor): The output from the MLP layer (after transformation).

        Returns:
            float: The computed parametric knowledge score.
        """
        # Move to GPU
        mlp_inputs = mlp_inputs.to(device = self.llm.device)
        mlp_outputs = mlp_outputs.to(device = self.llm.device)
        
        # Calculate logits for the last token before and after MLP
        input_logits = self.llm.lm_head(self.llm.model.norm(mlp_inputs))
        output_logits = self.llm.lm_head(self.llm.model.norm(mlp_outputs))

        # Convert logits to probabilities
        input_probs = torch.nn.functional.softmax(input_logits, dim=-1).detach().float().cpu().numpy()
        output_probs = torch.nn.functional.softmax(output_logits, dim=-1).detach().float().cpu().numpy()
        jsd = jensenshannon(input_probs, output_probs, axis = -1)

        return jsd
    

    def generate(
        self,
        instructions: str,
        context: str,
        query: str
    ) -> dict[str, str | float]:
        """
        Generate text for a (context, instructions) pair, and compute
        external context scores and parametric knowledge scores for each generated token.

        Args:
            instructions (str): The instructions string.
            context (str): The context string.
            query (str): The query string.

        Returns:
            response_dict (dict): A dictionary containing:
                'response' (str): The generated text.
                'parametric_score' (float): The summed parametric knowledge score.
                'context_score' (float): The summed external context score.
        """
        (tokenized_prompt,
         instruction_token_indices,
         context_token_indices,
         query_token_indices) = tokenize(
            instructions, context, query, self.tokenizer
        )
        prompt_len = len(tokenized_prompt)
        k = min(self.top_k, len(context_token_indices))
        llm_device = self.llm.device
        tensor_device = self.tensor_device
        response_dict = {}

        with self.llm.generate(tokenized_prompt, **self.sampling_params) as tracer:
            response_tokens = torch.full(
                size = (self.max_new_tokens,),
                fill_value = self.tokenizer.pad_token_id, # Fill with pad token initially
                device = tensor_device,
                dtype=torch.long  # Token IDs should be integers
            ).save()

            with tracer.iter[:] as token_idx:
                # Compute response tokens - move to tensor device
                response_tokens[token_idx] = self.llm.logits[0, -1, :].argmax().to(tensor_device)


        response = self.llm.tokenizer.decode(response_tokens.cpu(), skip_special_tokens=True)

        response_dict = {
            "response": response,
        }

        # Explicitly delete large tensors to free memory after .save() references
        # This is important because .save() keeps references alive
        del response_tokens
        del tracer
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return response_dict
    

    def generate_full(
        self,
        instructions: str,
        context: str,
        query: str
    ) -> dict[str, str | float]:
        """
        Generate text for a (context, instructions) pair, and compute
        external context scores and parametric knowledge scores for each generated token.

        Args:
            instructions (str): The instructions string.
            context (str): The context string.
            query (str): The query string.

        Returns:
            response_dict (dict): A dictionary containing:
                'response' (str): The generated text.
                'parametric_score' (float): The summed parametric knowledge score.
                'context_score' (float): The summed external context score.
        """
        (tokenized_prompt,
         instruction_token_indices,
         context_token_indices,
         query_token_indices) = tokenize(
            instructions, context, query, self.tokenizer
        )
        prompt_len = len(tokenized_prompt)
        llm_device = self.llm.device
        tensor_device = self.tensor_device
        response_dict = {}

        with self.llm.generate(tokenized_prompt, **self.sampling_params) as tracer:
            attention_probabilities = torch.zeros(
                size = (self.max_new_tokens, self.n_layers, self.n_heads, prompt_len + self.max_new_tokens),
                device = tensor_device,
                dtype = torch.bfloat16
            ).save()

            mlp_inputs = torch.zeros(
                size = (self.max_new_tokens, self.n_layers, self.llm.config.hidden_size), device=tensor_device, dtype=torch.bfloat16
            ).save()

            mlp_outputs = torch.zeros(
                size = (self.max_new_tokens, self.n_layers, self.llm.config.hidden_size), device=tensor_device, dtype=torch.bfloat16
            ).save()

            context_embeddings = torch.zeros(
                (len(context_token_indices), self.llm.config.hidden_size), device=tensor_device, dtype=torch.bfloat16
            ).save()

            response_embeddings = torch.zeros(
                (self.max_new_tokens, self.llm.config.hidden_size), device=tensor_device, dtype=torch.bfloat16
            ).save()

            response_tokens = torch.full(
                size = (self.max_new_tokens,),
                fill_value = self.tokenizer.pad_token_id, # Fill with pad token initially
                device = tensor_device,
                dtype=torch.long  # Token IDs should be integers
            ).save()

            with tracer.iter[:] as token_idx:
                for layer_idx, layer in enumerate(self.llm.model.layers):
                    # Attention shape: [batch_size, num_heads, seq_len, seq_len] - move to tensor device
                    attention_probabilities[token_idx, layer_idx, :, :prompt_len + token_idx] = self.llm.attention_probabilities[layer_idx][-1,:,-1,:].detach().to(tensor_device)

                    # Cache MLP inputs and outputs for this layer - move to tensor device
                    mlp_inputs[token_idx, layer_idx, :] = self.llm.mlps_input[layer_idx][-1, -1, :].detach().to(tensor_device)
                    mlp_outputs[token_idx, layer_idx, :] = self.llm.mlps_output[layer_idx][-1, -1, :].detach().to(tensor_device)
                
                # Last layer embeddings for context and current token - move to tensor device
                if token_idx == 0:
                    context_embeddings[:,:] = self.llm.model.output.last_hidden_state[-1, context_token_indices, :].detach().to(tensor_device)

                response_embeddings[token_idx, :] = self.llm.model.output.last_hidden_state[-1, -1, :].detach().to(tensor_device)

                # Compute response tokens - move to tensor device
                response_tokens[token_idx] = self.llm.logits[0, -1, :].argmax().detach().to(tensor_device)


        response = self.llm.tokenizer.decode(response_tokens.cpu(), skip_special_tokens=True)

        response_dict = {
            "response": response,
        }

        n_generated = np.sum(response_tokens.cpu().numpy() != self.tokenizer.pad_token_id)

        # External context scores
        response_dict['context_scores'] = self.compute_external_context_score(
            response_embeddings[:n_generated, :],
            context_embeddings,
            context_token_indices,
            attention_probabilities[:n_generated],
            prompt_len
        )

        # Parametric knowledge scores
        response_dict['parametric_scores'] = self.compute_parametric_knowledge_score(
            mlp_inputs[:n_generated, :, :],
            mlp_outputs[:n_generated, :, :]
        )

        # Explicitly delete large tensors to free memory after .save() references
        # This is important because .save() keeps references alive
        del mlp_inputs, mlp_outputs, attention_probabilities
        del context_embeddings, response_embeddings, response_tokens
        del tracer
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return response_dict

    
    def predict(
        self,
        prompts : list[tuple[str, str, str]],
        ids : list[str] = None
    ) -> tuple[list[str], list[float]]:
        """
        Generate text for a batch of (context, instructions) pairs, and compute
        a hallucination score for each generation.

        Args:
            prompts (list[tuple[str, str, str]]): A list of (instructions, context, query) pairs.
        
        Returns:
            responses (list[dict]): A list of dictionaries containing:
                'response' (str): The generated text.
                'parametric_score' (float): The summed parametric knowledge score.
                'context_score' (float): The summed external context score.
        """
        responses = []
        for i, (instructions, context, query) in enumerate(tqdm(prompts)):
            if self.generate_full_output:
                response_dict = self.generate_full(instructions, context, query)
            else:
                response_dict = self.generate(instructions, context, query)

            responses.append(response_dict)

        return responses
    

    def save(
        self,
        path : str
    ):
        """
        Save the recorded responses, parametric scores, and context scores to a .npz file.

        Args:
            path (str): The file path to save the data to.
        """
        np.savez(
            path,
            responses = self.responses,
            parametric_scores = np.array(self.parametric_score_arrays),
            context_scores = np.array(self.context_score_array)
        )


