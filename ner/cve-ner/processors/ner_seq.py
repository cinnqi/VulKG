""" Named entity recognition fine-tuning: utilities to work with CLUENER task. """
import torch
import logging
import os
import copy
import json
from processors.utils_ner import DataProcessor
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, input_len,segment_ids, label_ids, predict_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len
        self.predict_mask = predict_mask

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels, all_predict_mask = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:,:max_len]
    all_predict_mask = all_predict_mask[:, :max_len]
    # print(max_len)
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels,all_lens, all_predict_mask

def convert_examples_to_features(examples,label_list,max_seq_length,tokenizer,
                                 cls_token_at_end=False,cls_token="[CLS]",cls_token_segment_id=1,
                                 sep_token="[SEP]",pad_on_left=False,pad_token=0,pad_token_segment_id=0,
                                 sequence_a_segment_id=0,mask_padding_with_zero=True,):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    import tqdm
    for (ex_index, example) in tqdm.tqdm(enumerate(examples)):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        add_label = 'X'
        tokens = []
        predict_mask = []
        label_ids = []
        # print(len(example.labels))
        # print(len(example.text_a))
        # print('--------')
        if ex_index == 1:
            print(example.text_a)
            print(len(example.text_a))
            print(example.labels)
            print(len(example.labels))
        for i,word in enumerate(example.text_a):
            # print(example.text_a)
            sub_words = tokenizer.tokenize(word)
            # tokens = tokenizer.tokenize(example.text_a)
            if not sub_words:
                sub_words = ['[UNK]']
            tokens.extend(sub_words)

            for j in range(len(sub_words)):
                if j == 0:
                    predict_mask.append(1)
                    # print(example.labels[i],ex_index,word)
                    label_ids.append(label_map[example.labels[i]])
                else:
                    # '##xxx' -> 'X' (see bert paper)
                    predict_mask.append(0)
                    label_ids.append(label_map[add_label])
        if ex_index == 1:
            print(label_ids)

        # print("label_ids:{}".format(len(label_ids)))
        # label_ids = [label_map[x] for x in example.labels]
        # Account for [CLS] and [SEP] with "- 2".
        # if len(tokens)>510:
        #     print(tokens)
        #     print(len(tokens))
        #     print(label_ids)
        #     print(len(label_ids))

        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            predict_mask = predict_mask[: (max_seq_length - special_tokens_count)]
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [label_map['O']]
        predict_mask += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [label_map['O']]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [label_map['O']] + label_ids
            predict_mask = [0] + predict_mask
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(label_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            # 对于补齐的给它pad的id作为label_id，这个无所谓，因为有input_mask,为pad专门设一个label也可以理解，因为pad很多
            label_ids += [pad_token] * padding_length
            predict_mask += [0] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(predict_mask) == max_seq_length

        # print(max_seq_length)
        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s", example.guid)
        #     logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        #     logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask,input_len = input_len,
                                      segment_ids=segment_ids, label_ids=label_ids, predict_mask=predict_mask))
    return features


# class CnerProcessor(DataProcessor):
#     """Processor for the chinese ner data set."""
#
#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_text(os.path.join(data_dir, "train.char.bmes")), "train")
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_text(os.path.join(data_dir, "dev.char.bmes")), "dev")
#
#     def get_test_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_text(os.path.join(data_dir, "test.char.bmes")), "test")
#
#     def get_labels(self):
#         """See base class."""
#         return ["X",'B-CONT','B-EDU','B-LOC','B-NAME','B-ORG','B-PRO','B-RACE','B-TITLE',
#                 'I-CONT','I-EDU','I-LOC','I-NAME','I-ORG','I-PRO','I-RACE','I-TITLE',
#                 'O','S-NAME','S-ORG','S-RACE',"[START]", "[END]"]
#
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             if i == 0:
#                 continue
#             guid = "%s-%s" % (set_type, i)
#             text_a= line['words']
#             # BIOS
#             labels = []
#             for x in line['labels']:
#                 if 'M-' in x:
#                     labels.append(x.replace('M-','I-'))
#                 elif 'E-' in x:
#                     labels.append(x.replace('E-', 'I-'))
#                 else:
#                     labels.append(x)
#             examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
#         return examples

class CluenerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["X", 
                "B-cause", "B-location", "B-consequence", 'B-related issue', 'B-triggering operation', 'B-version',
                'B-attacker', 'B-happened scenario',
                "I-cause", "I-location", "I-consequence", 'I-related issue', 'I-triggering operation', 'I-version',
                'I-attacker', 'I-happened scenario',
                # "E-cause", "E-location", "E-consequence", 'E-related issue', 'E-triggering operation', 'E-version',
                # 'E-attacker', 'E-happened scenario',
                # "S-cause", "S-location", "S-consequence", 'S-related issue', 'S-triggering operation', 'S-version',
                # 'S-attacker', 'S-happened scenario',
                'O',"[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        # print(33333)
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
            #print(examples[0])
        return examples

ner_processors = {
    # "cner": CnerProcessor,
    'cve':CluenerProcessor
}
