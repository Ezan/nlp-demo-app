from functools import reduce
from uuid import uuid1
from logging import raiseExceptions
from typing_extensions import final
from nltk import tokenize
import torch
import tensorflow as tf
import math
import os
import numpy as np
import copy
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import collections
from collections.abc import Iterable

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    SquadFeatures,
    BasicTokenizer,
    squad_convert_examples_to_features,
    pipeline
)
from transformers.data.processors.squad import SquadExample, SquadResult

class CoreEngine:
    def __init__(self, model_path=None, use_fast=True):
        self.min_threshold = 0.00002
        self.max_context_sentences = 10
        self.max_seq_length = 384
        self.doc_stride = 128
        self.n_best_size = 50
        self.max_query_length = 64
        self.max_answer_length = 25
        self.do_lower_case = False
        self.null_score_diff_threshold = 0.0
        self.model_path = os.path.join(os.getcwd(),'web_integration\\engine\\model\\roberta_second_iter') if model_path == None else model_path
        config_class, model_class, tokenizer_class = (
            AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer)
        self.config = config_class.from_pretrained(self.model_path)
        self.tokenizer = tokenizer_class.from_pretrained(self.model_path, use_fast=use_fast)
        self.model = model_class.from_pretrained(self.model_path, config=self.config)


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _to_list(self, tensor):
        return tensor.detach().cpu().tolist()

    def _score_filter(self, positions):
        items = copy.deepcopy(positions)
        intermediate = []
        finalist = []
        while(len(items) > 0):
            first = items.pop(0)
            similar = [first] + list(filter(lambda x: x[0] == first[0], items))
            items = list(filter(lambda x: x[0] != first[0], items))
            similar = max(similar, key=lambda x:abs(x[0]-x[1]))
            intermediate.append(similar)
        while (len(intermediate) > 0):
            fp = intermediate.pop(0)
            finalist.append(fp)
            intermediate = list(filter(lambda x: (fp[0] > x[0] and fp[0] > x[1]) or (fp[0] < x[0] and fp[1] < x[1]),intermediate))
        return finalist
    
    def _sentChunker(self, list):
        for i in range(0, len(list), self.max_context_sentences):
            yield list[i: i+self.max_context_sentences]
       
    def _preliminaryContextSplit(self, question, context):
        sents = tokenize.sent_tokenize(context)
        contexts = list(self._sentChunker(sents))
        contexts = list(map(lambda x: ' '.join(x), contexts))
        questions = [question for _ in range(0,len(contexts))]
        return questions, contexts

    def _initFastExamples(self, question, context):
        if isinstance(question, list) and isinstance(context, str):
            inputs = [{"question": Q, "context": context} for Q in question]
        elif isinstance(question, list) and isinstance(context, list):
            if len(question) != len(context):
                raise ValueError("Questions and contexts don't have the same lengths")

            inputs = [{"question": Q, "context": C} for Q, C in zip(question, context)]
        elif isinstance(question, str) and isinstance(context, str):
            # ADD CHUNKING TO CONTEXT
            try:
                questions, contexts = self._preliminaryContextSplit(question, context)
            except:
                questions, contexts = [question], [context]

            inputs = [{"question": q, "context": c} for q, c in zip(questions, contexts)]
        else:
            raise ValueError("Arguments can't be understood")
        
        # Normalize inputs
        if isinstance(inputs, dict):
            inputs = [inputs]
        elif isinstance(inputs, Iterable):
            # Copy to avoid overriding arguments
            inputs = [i for i in inputs]
        else:
            raise ValueError("Invalid arguments {}".format(inputs))

        for i, item in enumerate(inputs):
            inputs[i] = self._normalize(item)

        return inputs
        
    def __call__(self, question, context):
        examples = self._initFastExamples(question, context)
        all_answers = []
        if self.tokenizer.is_fast:
            features_list = []
            for example in examples:
                encoded_inputs = self.tokenizer(
                    text=example.question_text,
                    text_pair=example.context_text,
                    padding='longest',
                    truncation="only_second",
                    max_length=self.max_seq_length,
                    stride=self.doc_stride,
                    return_tensors="np",
                    return_token_type_ids=True,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    return_special_tokens_mask=True,
                )
                # "num_span" is the number of output samples generated from the overflowing tokens.
                num_spans = len(encoded_inputs["input_ids"])
                # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
                # We put 0 on the tokens from the context and 1 everywhere else (question and special tokens)
                p_mask = np.asarray(
                    [
                        [tok != 1 for tok in encoded_inputs.sequence_ids(span_id)]
                        for span_id in range(num_spans)
                    ]
                )

                # keep the cls_token unmasked (some models use it to indicate unanswerable questions)
                if self.tokenizer.cls_token_id:
                    cls_index = np.nonzero(encoded_inputs["input_ids"] == self.tokenizer.cls_token_id)
                    p_mask[cls_index] = 0

                features = []
                for span_idx in range(num_spans):
                    features.append(
                        SquadFeatures(
                            input_ids=encoded_inputs["input_ids"][span_idx],
                            attention_mask=encoded_inputs["attention_mask"][span_idx],
                            token_type_ids=encoded_inputs["token_type_ids"][span_idx],
                            p_mask=p_mask[span_idx].tolist(),
                            encoding=encoded_inputs[span_idx],
                            # We don't use the rest of the values - and actually
                            # for Fast tokenizer we could totally avoid using SquadFeatures and SquadExample
                            cls_index=None,
                            token_to_orig_map={},
                            example_index=0,
                            unique_id=0,
                            paragraph_len=0,
                            token_is_max_context=0,
                            tokens=[],
                            start_position=0,
                            end_position=0,
                            is_impossible=False,
                            qas_id=None,
                        )
                    )
                features_list.append(features)
            
            for features, example in zip(features_list, examples):

                model_input_names = self.tokenizer.model_input_names
                fw_args = {k: [feature.__dict__[k] for feature in features] for k in model_input_names}
                with torch.no_grad():
                    # Retrieve the score for the context tokens only (removing question tokens)
                    fw_args = {k: torch.tensor(v, device=self.device) for (k, v) in fw_args.items()}
                    # On Windows, the default int type in numpy is np.int32 so we get some non-long tensors.
                    fw_args = {k: v.long() if v.dtype == torch.int32 else v for (k, v) in fw_args.items()}
                    start, end = self.model(**fw_args)[:2]
                    start, end = start.cpu().numpy(), end.cpu().numpy()

                min_null_score = 1000000  # large and positive
                answers = []
                for (feature, start_, end_) in zip(features, start, end):
                    undesired_tokens = np.abs(np.array(feature.p_mask) - 1) & feature.attention_mask

                    # Generate mask
                    undesired_tokens_mask = undesired_tokens == 0.0

                    # Make sure non-context indexes in the tensor cannot contribute to the softmax
                    start_ = np.where(undesired_tokens_mask, -10000.0, start_)
                    end_ = np.where(undesired_tokens_mask, -10000.0, end_)

                    # Normalize logits and spans to retrieve the answer
                    start_ = np.exp(start_ - np.log(np.sum(np.exp(start_), axis=-1, keepdims=True)))
                    end_ = np.exp(end_ - np.log(np.sum(np.exp(end_), axis=-1, keepdims=True)))

                    min_null_score = min(float('inf'), (start_[0] * end_[0]).item())
                    # Mask CLS
                    start_[0] = end_[0] = 0.0

                    # Ensure we have batch axis
                    if start_.ndim == 1:
                        start_ = start_[None]

                    if end_.ndim == 1:
                        end_ = end_[None]

                    # Compute the score of each tuple(start, end) to be the real answer
                    outer = np.matmul(np.expand_dims(start_, -1), np.expand_dims(end_, 1))
                    # Remove candidate with end < start and end - start > max_answer_len
                    candidates = np.tril(np.triu(outer), self.max_answer_length - 1)
                    scores_flat = candidates.flatten()

                    # NUMBER 15 CAN BE EMPIRICALLY SET TO SOME OTHER NUMBERS
                    idx = np.argpartition(-scores_flat, 15)[0 : 15]
                    idx_sort = idx[np.argsort(-scores_flat[idx])]

                    start_cord, end_cord = np.unravel_index(idx_sort, candidates.shape)[1:]
                    try:
                        filtered = self._score_filter([p for p in zip(start_cord, end_cord)])
                    except Exception as e:
                        print(e)
                        filtered = None

                    if not (filtered == None or (isinstance(filtered, list) and len(filtered) == 0)):
                        start_cord_f, end_cord_f = zip(*filtered)
                    else:
                        start_cord_f = start_cord
                        end_cord_f = end_cord

                    answers.append({"score": min_null_score, "start": 0, "end": 0, "answer": ""})
                    candidates = candidates[0,start_cord_f, end_cord_f]
                    enc=feature.encoding
                    answers += [
                                            {
                                                "score": score.item(),
                                                "start": enc.word_to_chars(
                                                    enc.token_to_word(s), sequence_index=1
                                                )[0],
                                                "end": enc.word_to_chars(enc.token_to_word(e), sequence_index=1)[
                                                    1
                                                ],
                                                "answer": example.context_text[
                                                    enc.word_to_chars(enc.token_to_word(s), sequence_index=1)[
                                                        0
                                                    ] : enc.word_to_chars(enc.token_to_word(e), sequence_index=1)[
                                                        1
                                                    ]
                                                ],
                                            }
                                            for s, e, score in zip(start_cord_f, end_cord_f, candidates) if score.item() > self.min_threshold
                                        ]
                answers.append({"score": min_null_score, "start": 0, "end": 0, "answer": ""})
                answers = sorted(answers, key=lambda x: x["score"], reverse=True)
                all_answers += answers

        else :
            features, dataset = squad_convert_examples_to_features(
                        examples=examples,
                        tokenizer=self.tokenizer,
                        max_seq_length=self.max_seq_length,
                        doc_stride=self.doc_stride,
                        max_query_length=self.max_query_length,
                        padding_strategy='max_length',
                        is_training=False,
                        return_dataset='pt',
                        tqdm_enabled = False)

            eval_sampler = SequentialSampler(dataset)
            eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=10)

            all_results = []
            

            for batch in eval_dataloader:
                self.model.eval()
                batch = tuple(t.to(self.device) for t in batch)

                with torch.no_grad():
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                    }

                    example_indices = batch[3]

                    outputs = self.model(**inputs)

                    for i, example_index in enumerate(example_indices):
                        eval_feature = features[example_index.item()]
                        unique_id = int(eval_feature.unique_id)

                        output = [self._to_list(output[i]) for output in outputs.to_tuple()]

                        start_logits, end_logits = output
                        result = SquadResult(unique_id, start_logits, end_logits)
                        all_results.append(result)

            _, answers = self._compute_predictions_logits(
                all_examples=examples,
                all_features=features,
                all_results=all_results,
                n_best_size= self.n_best_size,
                max_answer_length = self.max_answer_length,
                do_lower_case=self.do_lower_case,
                version_2_with_negative=True,
                null_score_diff_threshold=self.null_score_diff_threshold,
                tokenizer=self.tokenizer
            )
            all_answers += sorted(list(map(lambda x: {'score':x['probability'], 'start': x['start_index'], 'end': x['end_index'], 'answer':x['text']}, list(reduce(lambda n,p: n + p, answers.values(), [])))),key=lambda k:k['score'], reverse=True)
            all_answers = list(filter(lambda x: x['score'] >= self.min_threshold, all_answers))
        return all_answers

    def _get_best_indexes(self, logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    def _compute_predictions_logits(
        self,
        all_examples,
        all_features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        version_2_with_negative,
        null_score_diff_threshold,
        tokenizer,
    ):

        example_index_to_features = collections.defaultdict(list)
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
        )

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        scores_diff_json = collections.OrderedDict()

        for (example_index, example) in enumerate(all_examples):
            features = example_index_to_features[example_index]

            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            score_null = 1000000  # large and positive
            min_null_feature_index = 0  # the paragraph slice with min null score
            null_start_logit = 0  # the start logit at the slice with min null score
            null_end_logit = 0  # the end logit at the slice with min null score
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                start_indexes = self._get_best_indexes(result.start_logits, n_best_size)
                end_indexes = self._get_best_indexes(result.end_logits, n_best_size)
                
                # #REMOVE CONSECUTIVE ANSWERS
                start_indexes, end_indexes = zip(* self._score_filter([(x,y) for x,y in zip(start_indexes, end_indexes)]))

                # if we could have irrelevant answers, get the min score of irrelevant
                if version_2_with_negative:
                    feature_null_score = result.start_logits[0] + result.end_logits[0]
                    if feature_null_score < score_null:
                        score_null = feature_null_score
                        min_null_feature_index = feature_index
                        null_start_logit = result.start_logits[0]
                        null_end_logit = result.end_logits[0]
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index >= len(feature.tokens):
                            continue
                        if end_index >= len(feature.tokens):
                            continue
                        if start_index not in feature.token_to_orig_map:
                            continue
                        if end_index not in feature.token_to_orig_map:
                            continue
                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue
                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=result.start_logits[start_index],
                                end_logit=result.end_logits[end_index],
                            )
                        )
            if version_2_with_negative:
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=min_null_feature_index,
                        start_index=0,
                        end_index=0,
                        start_logit=null_start_logit,
                        end_logit=null_end_logit,
                    )
                )
            prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

            _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                "NbestPrediction", ["text", "start_logit", "end_logit", "start_index", "end_index"]
            )

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= n_best_size:
                    break
                feature = features[pred.feature_index]
                if pred.start_index > 0:  # this is a non-null prediction
                    tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                    orig_doc_start = feature.token_to_orig_map[pred.start_index]
                    orig_doc_end = feature.token_to_orig_map[pred.end_index]
                    orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]

                    tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                    # tok_text = " ".join(tok_tokens)
                    #
                    # # De-tokenize WordPieces that have been split off.
                    # tok_text = tok_text.replace(" ##", "")
                    # tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)

                    final_text = self._get_final_text(tok_text, orig_text, do_lower_case)
                    if final_text in seen_predictions:
                        continue

                    seen_predictions[final_text] = True
                else:
                    final_text = ""
                    seen_predictions[final_text] = True

                nbest.append(_NbestPrediction(text=final_text , start_logit=pred.start_logit, end_logit=pred.end_logit, start_index=pred.start_index, end_index=pred.end_index))
            # if we didn't include the empty option in the n-best, include it
            if version_2_with_negative:
                if "" not in seen_predictions:
                    nbest.append(_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit))

                # In very rare edge cases we could only have single null prediction.
                # So we just create a nonce prediction in this case to avoid failure.
                if len(nbest) == 1:
                    nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_index=0, end_index=0))

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

            assert len(nbest) >= 1, "No valid predictions"

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if not best_non_null_entry:
                    if entry.text:
                        best_non_null_entry = entry

            probs = self._compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_index"] = entry.start_index
                output["end_index"] = entry.end_index
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                nbest_json.append(output)
            assert len(nbest_json) >= 1, "No valid predictions"

            if not version_2_with_negative:
                all_predictions[example.qas_id] = nbest_json[0]["text"]
            else:
                # predict "" iff the null score - the score of best non-null > threshold
                score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
                scores_diff_json[example.qas_id] = score_diff
                if score_diff > null_score_diff_threshold:
                    all_predictions[example.qas_id] = ""
                else:
                    all_predictions[example.qas_id] = best_non_null_entry.text
            all_nbest_json[example.qas_id] = nbest_json
        return all_predictions, all_nbest_json

    def _compute_softmax(self, scores):
        """Compute softmax probability over raw logits."""
        if not scores:
            return []

        max_score = None
        for score in scores:
            if max_score is None or score > max_score:
                max_score = score

        exp_scores = []
        total_sum = 0.0
        for score in scores:
            x = math.exp(score - max_score)
            exp_scores.append(x)
            total_sum += x

        probs = []
        for score in exp_scores:
            probs.append(score / total_sum)
        return probs
    
    def _create_sample(self, question, context):
        if isinstance(question, list):
            return [SquadExample(uuid1(), q, c, None, None, None) for q, c in zip(question, context)]
        else:
            return SquadExample(uuid1(), question, context, None, None, None)

    def _normalize(self, item):
        if isinstance(item, SquadExample):
            return item
        elif isinstance(item, dict):
            for k in ["question", "context"]:
                if k not in item:
                    raise KeyError("You need to provide a dictionary with keys {question:..., context:...}")
                elif item[k] is None:
                    raise ValueError("`{}` cannot be None".format(k))
                elif isinstance(item[k], str) and len(item[k]) == 0:
                    raise ValueError("`{}` cannot be empty".format(k))
            
            return self._create_sample(**item)

    def _get_final_text(self, pred_text, orig_text, do_lower_case):
        """Project the tokenized prediction back to the original text."""

        # When we created the data, we kept track of the alignment between original
        # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
        # now `orig_text` contains the span of our original text corresponding to the
        # span that we predicted.
        #
        # However, `orig_text` may contain extra characters that we don't want in
        # our prediction.
        #
        # For example, let's say:
        #   pred_text = steve smith
        #   orig_text = Steve Smith's
        #
        # We don't want to return `orig_text` because it contains the extra "'s".
        #
        # We don't want to return `pred_text` because it's already been normalized
        # (the SQuAD eval script also does punctuation stripping/lower casing but
        # our tokenizer does additional normalization like stripping accent
        # characters).
        #
        # What we really want to return is "Steve Smith".
        #
        # Therefore, we have to apply a semi-complicated alignment heuristic between
        # `pred_text` and `orig_text` to get a character-to-character alignment. This
        # can fail in certain cases in which case we just return `orig_text`.

        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for (i, c) in enumerate(text):
                if c == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(c)
            ns_text = "".join(ns_chars)
            return (ns_text, ns_to_s_map)

        # We first tokenize `orig_text`, strip whitespace from the result
        # and `pred_text`, and check if they are the same length. If they are
        # NOT the same length, the heuristic has failed. If they are the same
        # length, we assume the characters are one-to-one aligned.
        tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

        tok_text = " ".join(tokenizer.tokenize(orig_text))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            return orig_text

        # We then project the characters in `pred_text` back to `orig_text` using
        # the character-to-character alignment.
        tok_s_to_ns_map = {}
        for (i, tok_index) in tok_ns_to_s_map.items():
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            return orig_text

        output_text = orig_text[orig_start_position : (orig_end_position + 1)]
        return output_text

if __name__ == '__main__':
    question = "Find the mention of monetary amounts that should be reviewed"
    context =  '''I wanted to begin by reviewing our 2017 accomplishments in what was a very busy and productive year for Windstream as we position the company for the growth opportunities ahead. During the year, we completed the EarthLink and Broadview acquisitions, which have given us a meaningful advantage with significant strategic best-of-breed assets such as SD-WAN expertise, and OfficeSuite, Broadview's UCaaS platform.

We developed and launched SD-WAN Concierge, our flagship SD-WAN product offering, as well as OfficeSuite across our entire company footprint. We expanded our enterprise contribution margin percentage by 200 basis points sequentially and 160 basis points year-over-year. The company exited 2017 at its highest adjusted OIBDAR margin level since prior to the EarthLink acquisition. We delivered our 12th consecutive quarter of consumer ARPU growth in the fourth quarter. We significantly improved the maturity profile of our balance sheet by extending almost $2 billion of maturities out an average of more than two years. And our synergy plans remain on schedule, and will continue to ramp in 2018.

In summary, while we admittedly fell shy of our original full year adjusted OIBDAR guidance by less than 1%, it was a successful year for Windstream as we have positioned the company to be able to take full advantage of the meaningful opportunities in front of us and have improved financial and operating results for almost all metrics across the business. Two examples of this progress are shown in our increased penetration of broadband speeds and our growing SD-WAN sales.

Turning to slide 5. We increased our penetration of speeds of 25 meg or greater by another 300 basis points sequentially to 24%. During 2017, we more than doubled our penetration level of those higher speed tiers. By the end of 2018, we expect the penetration of 25 meg or greater speeds to be 36%. In addition, we enabled 74,000 homes for broadband or faster broadband under the FCC's Connect America Fund II program during the year, which keeps us on or ahead of schedule for all the program's deadlines and goals.

Growing demand for SD-WAN service continued and represented over 15% of total enterprise sales during the quarter, a metric that has increased every quarter throughout 2017. Our strategic enterprise sales, which include Unified Communications as a Service, SD-WAN and on-net sales, accounted for 38.4% of total enterprise sales in the quarter, up 230 basis points sequentially. By the end of 2018, we expect approximately 50% of our enterprise sales to be tied to our strategic products. I want to take this opportunity to personally thank all of our employees for their hard work and dedication throughout the past year.

'''
    model_path = os.path.join(os.getcwd(),'web_integration\\engine\\model\\roberta_second_iter')
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path, config=config)
    # result = pipeline("question-answering",model=model, tokenizer=tokenizer)(question=question,context=context,topk=5, handle_impossible_answer=True)    
    engine = CoreEngine(use_fast=True)
    contexts = context.split('.')
    result = engine(question, context)
    print(result)
