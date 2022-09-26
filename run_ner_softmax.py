import argparse
import glob
import logging
import os
import json

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from tools.common import seed_everything,json_to_text
from tools.common import init_logger, logger

from models.transformers import WEIGHTS_NAME,BertConfig,BertTokenizer
from models.bert_for_ner import BertSoftmaxForNer
from processors.utils_ner import CNerTokenizer,get_entities
from processors.ner_seq import convert_examples_to_features
from processors.ner_seq import ner_processors as processors
from processors.ner_seq import collate_fn
from metrics.ner_metrics import SeqEntityScore

os.environ["CUDA_VISIBLE_DEVICES"] ="0"

MODEL_CLASSES = {
    'bert': (BertConfig, BertSoftmaxForNer, BertTokenizer),
}


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=80,
                                                num_training_steps=675)

    # global_step作用其实是在保存模型的时候起一个tag作用
    # global_step = 0
    # current_epoch = 0

    # tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    best_f=0
    for epoch in range(int(args.num_train_epochs)):
        steps = 0
        total_loss = 0
        # pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            # if steps_trained_in_current_epoch > 0:
            #     steps_trained_in_current_epoch -= 1
            #     continue
            model.train()
            steps += 1
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            # print(inputs)
            # break
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            total_loss += loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
        print('epoch{} loss: {}'.format(epoch, total_loss/steps))
        if epoch > 0:
            eval_info = evaluate(args,model,tokenizer)
            eval_f = eval_info['f1']

            if eval_f>best_f:
                best_f = eval_f
                output_dir = os.path.join(args.output_dir, "best_checkpoint")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # 直接模型保存，如果分布式有另外表达方式
                model.save_pretrained(output_dir)
                # 保存定义的args参数
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                # 保存tokenizer，如果tokenizer改变了就加载保存的，而不是直接从s3加载
                tokenizer.save_vocabulary(output_dir)
                # 优化器和schdule保存
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                with open('best_epoch.txt','w',encoding='utf8') as f:
                    f.write(str(epoch))
        print("train finish")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    # return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix=""):
    # print(args.id2label)
    metric = SeqEntityScore(args.id2label,markup=args.markup)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    eval_dataset = load_and_cache_examples(args, args.task_name,tokenizer, data_type='dev')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)

    # eval_loss = 0.0
    # nb_eval_steps = 0
    # pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            input_id = inputs['labels'].cpu().numpy().tolist()[0]
            input_id = [args.id2label[x] for x in input_id]
            # print(input_id)
            # print(len(input_id))
            # print(inputs['input_ids'].shape)
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            # eval_loss += tmp_eval_loss.item()
        # nb_eval_steps += 1
        inputs['predict_mask'] = batch[-1]
        # print("logits_shape:{}".format(logits.shape))
        _, preds = torch.max(logits, dim=2, keepdim=False)
        # print(preds.shape)
        # print(inputs['labels'].shape)
        # print(inputs['predict_mask'])
        print(inputs)
        print(preds)
        print(len(inputs["labels"][0]))
        print(len(preds[0]))
        # print(inputs['predict_mask'].shape)
        valid_predicted = torch.masked_select(preds, inputs['predict_mask']==1)
        valid_label_ids = torch.masked_select(inputs['labels'], inputs['predict_mask']==1)
        # print(valid_predicted.shape)
        # print(valid_label_ids.shape)
        valid_label_ids = valid_label_ids.reshape(1, -1)
        valid_predicted = valid_predicted.reshape(1, -1)
        valid_label_ids = valid_label_ids.cpu().numpy().tolist()
        valid_predicted = valid_predicted.cpu().numpy().tolist()
        # print(len(valid_label_ids[0]))
        # print(len(valid_predicted[0]))
        # print(inputs['attention_mask'][0].shape)
        for i, label in enumerate(valid_label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                # if j == 0:
                #     continue
                # elif out_label_ids[i][j] == args.label2id['[SEP]']:
                temp_1.append(args.id2label[valid_label_ids[i][j]])
                temp_2.append(valid_predicted[i][j])
                if j == (len(valid_predicted[i])-1):
                    # print('@@@@@@@')
                    metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                # else:

        # pbar(step)
    # print(metric.origins)
    # print(metric.founds)
    print(' ')
    # eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    # results['loss'] = eval_loss
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info("***** Entity results %s *****", prefix)
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********"%key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)
    return results

def predict(args, model, tokenizer, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir):
        os.makedirs(pred_output_dir)

    test_dataset = load_and_cache_examples(args, args.task_name,tokenizer, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1,collate_fn=collate_fn)

    results = []
    # pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
        inputs['predict_mask'] = batch[-1]
        logits = outputs[0]
        _ ,preds= torch.max(logits, dim=2, keepdim=False)
        # print(inputs["predict_mask"])
        # print(preds)
        valid_predicted = torch.masked_select(preds, inputs['predict_mask']==1)
        # print(valid_predicted)
        preds = valid_predicted.detach().cpu().numpy().tolist()
        # preds = np.argmax(preds, axis=2).tolist()
        # preds = preds[0] # [CLS]XXXX[SEP]
        tags = [args.id2label[x] for x in preds]
        # 输入：[B-loc,I-loc,S-PER]
        # 输出：[[LOC,0,1],[PER,2,2]]
        label_entities = get_entities(preds, args.id2label, args.markup)
        json_d = {}
        json_d['id'] = step
        json_d['tag_seq'] = " ".join(tags)
        json_d['entities'] = label_entities
        results.append(json_d)
        # pbar(step)
    print(" ")
    output_predic_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
    output_submit_file = os.path.join(pred_output_dir, prefix, "test_submit.json")
    with open(output_predic_file, "w",encoding='utf8') as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')
    test_text = []
    with open(os.path.join(args.data_dir,"test.json"), 'r',encoding='utf8') as fr:
        for line in fr:
            test_text.append(json.loads(line))
    test_submit = []
    # 预测的结果转成json格式
    for x, y in zip(test_text, results):
        json_d = {}
        json_d['id'] = x['id']
        json_d['label'] = {}
        entities = y['entities']
        words = x['text']
        # 给它转成和原始的json格式相同，将源文本输入进去更好
        json_d['text'] = words
        if len(entities) != 0:
            for subject in entities:
                tag = subject[0]
                start = subject[1]
                end = subject[2]
                # word = "".join(words[start:end + 1])
                word = " ".join(words.split()[start:end+1])
                if tag in json_d['label']:
                    if word in json_d['label'][tag]:
                        json_d['label'][tag][word].append([start, end])
                    else:
                        json_d['label'][tag][word] = [[start, end]]
                else:
                    json_d['label'][tag] = {}
                    json_d['label'][tag][word] = [[start, end]]
        test_submit.append(json_d)
    json_to_text(output_submit_file,test_submit)

def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    processor = processors[task]()
    # 得到label_;ist
    label_list = processor.get_labels()
    if data_type == 'train':
        examples = processor.get_train_examples(args.data_dir)
    elif data_type == 'dev':
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)
    features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.train_max_seq_length if data_type=='train' \
                                                               else args.eval_max_seq_length,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                cls_token = tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_predict_mask = torch.tensor([f.predict_mask for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens,all_label_ids,all_predict_mask)

    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--data_dir",default=None,type=str,required=True,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",)
    parser.add_argument("--model_type",default=None,type=str,required=True,
                        help="Model type selected in the list: ")
    parser.add_argument("--model_path",default=None,type=str,required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " )
    parser.add_argument("--output_dir",default=None,type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.", )
    # Other parameters
    parser.add_argument('--markup',default='bio',type=str,choices=['bios','bio','bioes'])
    parser.add_argument('--loss_type', default='focal', type=str, choices=['lsr', 'focal', 'ce'])
    parser.add_argument( "--labels",default="",type=str,
                        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",)
    parser.add_argument( "--config_name", default="", type=str,
                         help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name",default="",type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name",)
    # parser.add_argument("--cache_dir",default="",type=str,
    #                     help="Where do you want to store the pre-trained models downloaded from s3", )
    parser.add_argument("--train_max_seq_length", default=512,type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--eval_max_seq_length",default=512,type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training",action="store_true",
                        help="Whether to run evaluation during training at each logging step.", )
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument( "--max_steps", default=-1,type=int,
                         help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)

    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints",action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",)
    parser.add_argument("--predict_checkpoints", type=int, default=0,
                        help="predict checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    init_logger(log_file=args.output_dir + '/{}.log'.format(args.model_type))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    seed_everything(args.seed)
    # 定义 NER task
    args.task_name = args.task_name.lower()
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    print(label_list)
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)
    print("len_labels:{}".format(num_labels))
    args.model_type = args.model_type.lower()
    # 加载模型，注意这里的Token和bert是基于继承原始模型的。token要加载save_pretrained才行，否则报错。
    # 如果是接着训练的话也可以加载保存的
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name,num_labels=num_labels,loss_type = args.loss_type)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    model = model_class.from_pretrained(args.model_path,config=config)

    model.to(args.device)
    print('model load finish')
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name,tokenizer, data_type='train')
        train(args, train_dataset, model, tokenizer)
    # Evaluation
    results = {}
    if args.do_eval:

        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)

        checkpoints = [os.path.join(args.output_dir, path) for path in os.listdir(args.output_dir)]
        checkpoints = [path for path in checkpoints if os.path.isdir(path)]
        print(f"checkpoints:{checkpoints}")

        for checkpoint in checkpoints:
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer)

            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
        checkpoints = [os.path.join(args.output_dir, path) for path in os.listdir(args.output_dir)]
        checkpoints = ["C:/Users/I572410/Desktop/Cinqi/python_project/CVE_TEST/ner/cve-ner/outputs/cve_softmax/bert/best_checkpoint"]

        for checkpoint in checkpoints:
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            predict(args, model, tokenizer)

if __name__ == "__main__":
    main()
