#cstm_generate.py

import torch
from torch.nn import functional as F

def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def enforce_repetition_penalty_(lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
        """
        Enforce the repetition penalty (from the `CTRL paper <https://arxiv.org/abs/1909.05858>`__).
        """
        for i in range(batch_size * num_beams):
            for previous_token in set(prev_output_tokens[i].tolist()):
                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if lprobs[i, previous_token] < 0:
                    lprobs[i, previous_token] *= repetition_penalty
                else:
                    lprobs[i, previous_token] /= repetition_penalty


def postprocess_next_token_scores (
        scores,
        input_ids,
        cur_len,
        min_length,
        max_length,
        repetition_penalty,
        batch_size,
        no_repeat_ngram_size=0,
        eos_token_id=None,
        bad_words_ids=None,
        num_beams=1,
    ):
        # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            enforce_repetition_penalty_(
                scores,
                batch_size,
                num_beams,
                input_ids,
                repetition_penalty,
            )

        # set eos token prob to zero if min_length is not reached
        #if eos_token_id is not None and cur_len < min_length:
            #scores[:, eos_token_id] = -float("inf")

        #if no_repeat_ngram_size > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            #num_batch_hypotheses = batch_size * num_beams
            # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
           # banned_batch_tokens = calc_banned_ngram_tokens(
                #input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
            #)
            #for i, banned_tokens in enumerate(banned_batch_tokens):
               # scores[i, banned_tokens] = -float("inf")

        if bad_words_ids is not None:
            # Exclude EOS token (already processed)
            bad_words_ids = list(filter(lambda bad_token_seq: bad_token_seq != [eos_token_id], bad_words_ids))
            # calculate a list of banned tokens according to bad words
            banned_tokens = calc_banned_bad_words_ids(input_ids.tolist(), bad_words_ids)
            # Modify the scores in place by setting the banned tokens logits to `-inf`
            set_scores_to_inf_for_banned_tokens(scores, banned_tokens)

        return scores


def generate_no_beam_search(model,
    input_ids,
    cur_len,
    max_length,
    min_length,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    batch_size,
    pad_token_id=None,
    eos_token_id=None,
    attention_mask=None,
    bad_words_ids=None,
    sess_type="transformer",
    create_batches=True
  ):
  """Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly. Adapted from huggingface transformers
        """

  if create_batches:
      input_ids=torch.cat([input_ids for _ in range(batch_size)], dim=0)
  #length of generated sentences / unfinished sentences
  unfinished_sents = input_ids.new(batch_size).fill_(1)
  sent_lengths = input_ids.new(batch_size).fill_(max_length)

  past = None
  while cur_len < max_length:
    #model_inputs = self.prepare_inputs_for_generation(
                  #input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs


    if sess_type=="transformer":
      #setting model inputs, possible speed up: "past_key_values": past, but don't
      #pass input ids for precomputed states, also must pass use_cache=True to LMHeadModel
      model_inputs={"input_ids": input_ids, "attention_mask": attention_mask}
      #must return dict so can slice logits
      outputs=model(**model_inputs)
      #slicing next token out of output. Output shape is batch size, tokens, vocab size
      next_token_logits = outputs.logits[:, -1, :]
      # if model has past, then set the past variable to speed up decoding
      if "past_key_values" in outputs:
        past = outputs.past_key_values
      elif "mems" in outputs:
        past = outputs.mems
      # extend attention_mask for new generated input if only decoder
      if True:
        attention_mask = torch.cat(
        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                  )

    elif sess_type=="onnx":
      model_inputs=input_ids.numpy()
      #selecting input and output variables for onnx
      model_in_var=model.get_inputs()[0].name
      model_out_var=model.get_outputs()[0].name

      outputs = model.run([model_out_var], {model_in_var: model_inputs})
      #stores output in list
      outputs = torch.tensor(outputs[0])
      next_token_logits = outputs[:, -1, :]


    #processing for repitition penalty and bad words
    scores = postprocess_next_token_scores(
                  scores=next_token_logits,
                  input_ids=input_ids,
                  cur_len=cur_len,
                  min_length=min_length,
                  max_length=max_length,
                  eos_token_id=eos_token_id,
                  repetition_penalty=repetition_penalty,
                  batch_size=batch_size,
                  num_beams=1,
              )



    #apply temperature
    if temperature != 1.0:
      scores = scores / temperature
    #apply top_k, top_p
    next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
    #apply softmax
    probs = F.softmax(next_token_logscores, dim=-1)
    #choose tokens
    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

    # update generations and finished sentences
    if eos_token_id is not None:
      # pad finished sentences if eos_token_id exist
      tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
    else:
      tokens_to_add = next_token
    # add token and increase length by one
    input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
    cur_len = cur_len + 1

    if eos_token_id is not None:
      eos_in_sents = tokens_to_add == eos_token_id
      # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
      is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
      sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
      # unfinished_sents is set to zero if eos in sentence
      unfinished_sents.mul_((~eos_in_sents).long())

    # stop when there is a </s> in each sentence, or if we exceed the maximul length
    if unfinished_sents.max() == 0:
      break

  return input_ids
