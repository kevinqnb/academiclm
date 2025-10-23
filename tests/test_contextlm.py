import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import functional as F

from scholarlm import ContextLM
from scholarlm.utils import tokenize

####################################################################################################
"""
ReDEeP-ICLR implementation of ContextLM v2 for testing purposes.
Original repo:
https://github.com/Jeryi-Sun/ReDEeP-ICLR/blob/main/ReDeEP/token_level_detect.py

NOTE: We have changed very little from the original implementation, except how the tokenization 
process is handled, and how the context tokens are viewed within the attention scores.
"""

def calculate_dist(sep_vocabulary_dist, sep_attention_dist):
    softmax_mature_layer = F.softmax(sep_vocabulary_dist, dim=-1)  
    softmax_anchor_layer = F.softmax(sep_attention_dist, dim=-1)  

    M = 0.5 * (softmax_mature_layer + softmax_anchor_layer) 

    # 4. Calculate log-softmax for the KL divergence
    log_softmax_mature_layer = F.log_softmax(sep_vocabulary_dist, dim=-1)  
    log_softmax_anchor_layer = F.log_softmax(sep_attention_dist, dim=-1) 

    # 5. Calculate the KL divergences and then the JS divergences
    kl1 = F.kl_div(log_softmax_mature_layer, M, reduction='none').mean(-1)  
    kl2 = F.kl_div(log_softmax_anchor_layer, M, reduction='none').mean(-1)  
    # # Fix bug: https://github.com/Jeryi-Sun/ReDEeP-ICLR/issues/2 but for stable calculation, we maintain the original implementation of JSD.
    # kl1 = F.kl_div(M.log(), softmax_mature.unsqueeze(0),  reduction='none').mean(-1)
    # kl2 = F.kl_div(M.log(), softmax_anchor,  reduction='none').mean(-1)
    js_divs = 0.5 * (kl1 + kl2) 
        
    return js_divs.cpu().item()*10e5


def contextlm_test(
    model_name: str,
    instructions: str,
    context: str,
    query: str,
    top_k: int = 10
):
    """
    ReDEeP-ICLR implementation of ContextLM v2 for testing purposes.
    Please see the original repo, and note that this code is attributed to those authors:
    https://github.com/Jeryi-Sun/ReDEeP-ICLR/blob/main/ReDeEP/token_level_detect.py

    NOTE: We have changed very little from the original implementation, except how the tokenization 
    process is handled, and how the context tokens are viewed within the attention scores.

    NOTE: This function computes the scores for only a single next token generation step.

    Args:
        instructions (str): The instructions string.
        context (str): The context string.
        query (str): The query string.
    Returns:
        external_context_score (float): The external context score.
        parameter_knowledge_score (float): The parameter knowledge score.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.float16,
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda"

    prompt, instruction_tokens, context_tokens, query_tokens = tokenize(
        instructions,
        context,
        query,
        tokenizer,
    )

    with torch.no_grad():
        outputs = model(
            input_ids=torch.tensor([prompt]).to(device),
            return_dict=True,
            output_attentions=True,
            output_hidden_states=True,
        )
    #logits_dict = {key: [value[0].to(device), value[1].to(device)] for key, value in logits_dict.items()}

    logits = outputs['logits']  # [batch, seq_len, vocab_size]
    attentions = outputs['attentions']  # tuple ([batch, num_heads, seq_len, seq_len], ..., )
    hidden_states = outputs["hidden_states"] # tuple ([batch, seq_len, vocab_size], ..., ) 
    last_hidden_states = hidden_states[-1][0, :, :] # [prefix_len, hidden_size]


    # todo 修改成 筛选 teacher focusing 的 token 和 model generate token 是否在 top_10内
    # probs = outputs['logits'][range(outputs["logits"].shape[0]), continue_ids].sum().item()
    # # ---------------------------------------------------------------------------------------------------------------
    external_similarity = [] # 这个用来存储生成的 token embedding 和 copy head 关注的 token embedding 的相似度得分
    parameter_knowledge_difference = []
    hallucination_label = []
    # 计算一下输入的 context 里面有没有 hallucination 词，如果有的话 copy 的时候把他们的 pointer weight 调小
    # input: input_ids, corr token vocab distribution
    # output: hallucination score for the input_ids or hallucination mask
    # outputs.attentions is a tuple, taking the last layer's attentions
    attentions_list = []
    for attentions_layer_id in range(len(attentions)):
        for head_id in range(attentions[attentions_layer_id].shape[1]):
            #if [attentions_layer_id, head_id] not in copy_heads:
            #    continue
            attentions_list.append({"layer_head":(attentions_layer_id, head_id), "attention_score":attentions[attentions_layer_id][:,head_id,:,:]}) 


    # Step 2: Extract the non-zero values from the last row/column
    # Now we gather the attention scores for the last token of each sequence
    pointer_scores_list = [attention_dict["attention_score"][:, -1, context_tokens] for attention_dict in attentions_list] # shape: (batch_size, sequence_length)
    pointer_probs_list = torch.cat(pointer_scores_list, dim=0)  # shape: (head_num, sequence_length)

    # Step 4: select the top attented token
    # Create an extended attention mask that masks out special tokens
    # hyperparameter: token rate

    # pointer_probs_list 是每个位置对应的大小(head_num, seq_len)，last_hidden_states shape (seq_len, hidden_state)是每个位置对应的 value，请取出 top 10% input_ids_cp 的 last_hidden_states，最终输出为(head_num, top10_len, hidden_state)
    # 获取top 10%的索引

    # 获取排序后的索引，按照概率从大到小排序
    sorted_indices = torch.argsort(pointer_probs_list, dim=1, descending=True)

    # 选择前top_k个索引
    top_k_indices = sorted_indices[:, :top_k]

    print("Context Top Indices:")
    print(top_k_indices)

    # 我们需要将 top_k_indices 展平，以便用于索引 last_hidden_states
    flattened_indices = top_k_indices.flatten()  # shape (head_num * k,)
    # 使用展平的索引在 last_hidden_states 中查找相应的 hidden_state
    selected_hidden_states = last_hidden_states[context_tokens]
    selected_hidden_states = selected_hidden_states[flattened_indices]  # shape (head_num * k, hidden_state)

    # 重新 reshape 成 (head_num, k, hidden_state)
    top_k_hidden_states = selected_hidden_states.view(top_k_indices.shape[0], top_k_indices.shape[1], -1)

    attend_token_hidden_state = torch.mean(top_k_hidden_states, dim=1) # (head_num, hidden_state)
    print("Mean attended Token Hidden State:")
    print(attend_token_hidden_state)

    print("Context Embeddings")
    print(last_hidden_states[context_tokens, :])

    print("Context Embedding Mean")
    print(torch.mean((last_hidden_states[context_tokens, :])[[6,7,5,1,2],:], dim=0))

    print("Last Token Embedding")
    print(last_hidden_states[-1, :])

    #print("Last hidden state:")
    #print(outputs.last_hidden_state)

    # Step 5: Calculate the similarity between the last token and the attentioned prefix text
    current_hidden_state = last_hidden_states[-1, :] # shape (hidden_state,)

    # 扩展 current_hidden_state 的形状以匹配 pointer_probs_list
    current_hidden_state = current_hidden_state.unsqueeze(0).expand(attend_token_hidden_state.shape)

    # 计算余弦相似度
    cosine_similarity = F.cosine_similarity(attend_token_hidden_state.to(device), current_hidden_state.to(device), dim=1)
    external_similarity.append(cosine_similarity.cpu().tolist())
    #parameter_knowledge_difference.append([calculate_dist(value[0][0,-1,:], value[1][0,-1,:]) for value in logits_dict.values()])

    external_context_score = cosine_similarity.sum().cpu().item()

    torch.cuda.empty_cache()

    return external_context_score, None


####################################################################################################


def test_simple_prompt():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    top_k = 5
    instructions = "Answer the question based on the context."
    context = "The color of the sky is purple today."
    query = "What is the color of the sky?"
    ecs, pks = contextlm_test(
        model_name,
        instructions,
        context,
        query,
        top_k=top_k
    )

    ctx_lm = ContextLM(
        model_name=model_name,
        top_k=top_k,
        sampling_params={"max_new_tokens": 1},
        nnsight_kwargs={"device_map": "auto", "dtype": torch.float16, "attn_implementation": "eager"},
    )
    responses = ctx_lm.predict([(
        instructions,
        context,
        query
    )])
    ecs_test = responses[0]['context_score']
    pcs_test = responses[0]['parametric_score']
    pcs_test = None

    assert abs(ecs - ecs_test) < 1e-5, f"External context scores do not match: {ecs} vs {ecs_test}"
    assert pks == pcs_test, f"Parameter knowledge scores do not match: {pks} vs {pcs_test}"


if __name__ == "__main__":
    test_simple_prompt()