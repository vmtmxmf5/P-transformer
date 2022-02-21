import sentencepiece as spm
import json
import torch
import torch.nn as nn
import time
import os
import numpy as np

from torch.utils.data import Dataset, DataLoader
from Ptransformer import Ptransformer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import logging
from transformers import get_scheduler
from accelerate import Accelerator, DistributedDataParallelKwargs

# from Ptransformer.py import *


class WMT_Dataset(nn.Module):
    def __init__(self, src_path, tgt_path, src_tokenizer, tgt_tokenizer):
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        # 반드시 rb로 읽을 것. 'r'로 읽으면 utf 맞춰도 line개수가 달라짐
        # vim으로 보는 line index랑 python으로 읽어온 index랑 달라지니 주의!
        with open(src_path, 'rb') as f, open(tgt_path, 'rb') as g:
            self.src_data = f.readlines()
            self.tgt_data = g.readlines()
    def __len__(self):
        return len(self.src_data)
    def __getitem__(self, index):
        src = self.src_data[index].decode('utf-8')
        tgt = self.tgt_data[index].decode('utf-8')
        src = torch.LongTensor([self.src_tokenizer.encode_as_ids(src)])
        tgt = torch.LongTensor([[self.tgt_tokenizer.bos_id()] + self.tgt_tokenizer.encode_as_ids(tgt) + [self.tgt_tokenizer.eos_id()]])
        src_len = src.shape[1]
        tgt_len = tgt.shape[1]
        return src, tgt, src_len, tgt_len


def WMT_collate(batch):
    srcs, tgts, src_len, tgt_len = zip(*batch)
    max_src, max_tgt = max(src_len), max(tgt_len)

    source, target = [], []
    for i, (src, tgt) in enumerate(zip(srcs, tgts)):
        pad_src = max_src - list(src_len)[i]
        pad_tgt = max_tgt - list(tgt_len)[i]
        src_tensor = torch.cat([src, torch.LongTensor([[1] * pad_src])], dim=1)
        tgt_tensor = torch.cat([tgt, torch.LongTensor([[1] * pad_tgt])], dim=1)
        source.append(src_tensor)
        target.append(tgt_tensor)

    sources = torch.cat(source, dim=0)
    targets = torch.cat(target, dim=0)
    return sources, targets, torch.LongTensor(src_len)


def train(model, optimizer, criterion, dataloader, pad_id, train_begin, epoch, device, accelerator):
    model.train()
    begin = epoch_begin = time.time()
    print_batch = 100
    total_num, total_batch_size = 0, len(dataloader)

    losses, batch = 0, 0
    print('train start...')

    for src, tgt, lengths in dataloader:
        # if src.shape[1] < 512 and tgt.shape[1] < 512:
#             src = src.to(device)
#             tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]

        # (Batch, T_dec, len(vocab))
        outputs = model(src, tgt_input, lengths)

        # + 파이토치는 backward path 타고 오면서 누적합된 그라디언트를 사용한다 (for RNN)
        # steps 마다 zero grad로 바꿔주지 않으면 이전 step의 그라디언트를 재활용하게 되고
        # 그건 loss를 min(max)하는 방향과는 다른 방향으로 update를 이끌 수 있다
        optimizer.zero_grad()

        # (Batch * T_dec)
        tgt_out = tgt[:, 1:].reshape(-1)
        # (Batch * T_dec, len(vocab))
        outputs = outputs.reshape(-1, outputs.shape[-1])
        # 실험해보니 loss는 input 2차원, target 1차원을 입력으로 받았음 (그래서 reshape 해야 함)
        loss = criterion(outputs, tgt_out)

        # https://tutorials.pytorch.kr/beginner/blitz/autograd_tutorial.html
        accelerator.backward(loss) # 계산 그래프 타고 부모 노드에서 자식노드로 그라디언트 계산(compute) 시작
        ## 이 때 requires_grad=True인 텐서에 대해서만 gradient of loss 계산해서 parameter.grad에 저장한다

        # grad_norm? gradient가 너무 커지면 gradient exploding일어날 수 있으니 그 경우 통제
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step() # parameter tensor의 .grad라는 attribute에 저장되어 있는 그라디언트를 꺼내서 옵티마이저 업데이트
        lr_scheduler.step()

        losses += loss.item()
        total_num += 1  

        if accelerator.is_local_main_process:
            if batch % print_batch == 0:
                current = time.time()
                elapsed = current - begin
                epoch_elapsed = (current - epoch_begin) / 60.0
                train_elapsed = (current - train_begin) / 3600.0

                print('epoch: {:4d}, batch: {:5d}/{:5d}, lr: {:.16f},\nloss: {:.8f}, elapsed: {:6.2f}s {:6.2f}m {:6.2f}h'.format(
                        epoch, batch, total_batch_size,
                        optimizer.param_groups[0]['lr'],
                        losses / total_num,
                        elapsed, epoch_elapsed, train_elapsed))
                begin = time.time()

        batch += 1
    print('train completed...')
    return losses / total_batch_size


def evaluate(model, criterion, dataloader, pad_id, tgt_tokenizer, device):
    losses = 0
    total_num, total_batch_size = 0, 0

    model.eval() # Dropout, BatchNorm 종료
    
    epoch_BLEU = []
    print('validation start...')
    with torch.no_grad(): # require_grads=False로 변경 -> 계산에 쓰이는 메모리양 감소
        for src, tgt, lengths in dataloader:
           # if src.shape[1] < 100 and tgt.shape[1] < 100:
#             src = src.to(device)
#             tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]

            # (Batch, T_dec, len(vocab))
            outputs = model(src, tgt_input, lengths)
            # y_hat = outputs.max(-1)[1]
            
            # (Batch * T_dec)
            tgt_out = tgt[:, 1:].reshape(-1)
            # (Batch * T_dec, len(vocab))
            outputs = outputs.reshape(-1, outputs.shape[-1])

            loss = criterion(outputs, tgt_out)
            losses += loss.item()
            total_num += 1 

            ################################################################
            ### BLEU Score 계산 Inference ###
            predictions, references = [], []
            
            chencherry = SmoothingFunction()
            for sample, label, leng in zip(src, tgt[:, 1:-1], lenghts):
                # (Tdec)
                token = model.search(sample, leng, max_length=130)
                prediction = tgt_tokenizer.decode_ids(token).split()
                reference = tgt_tokenizer.decode_ids(label.tolist()).split()
                # print('pred : ', prediction, '\n', 'ref : ', reference)
                predictions.append(prediction)
                references.append(reference)
                # print('bleu : ', sentence_bleu([reference], prediction))
            BLEU = corpus_bleu([references], predictions, smoothing_function=chencherry.method4)
            # print(BLEU)
            epoch_BLEU.append(BLEU)
    print('validation completed...')
    BLEU_score = np.mean(epoch_BLEU)
    return losses / total_num, BLEU_score
    

def get_logger(name: str, file_path: str, stream=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 시간, 로거 이름, 로깅 레벨, 메세지
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    # console에 출력하도록 설정
    stream_handler = logging.StreamHandler()
    # 현재 디렉토리에 파일로 로깅하도록 설정
    file_handler = logging.FileHandler(file_path)

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if stream:
        logger.addHandler(stream_handler)
    # 현재 디렉토리에 로깅 저장
    logger.addHandler(file_handler)

    return logger


def save(filename, model, logger, accelerator):
    save_model = accelerator.unwrap_model(model)
    state = {
        'model': save_model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    accelerator.save(state, filename)
    logger.info('Model saved')


def load(filename, model, optimizer, logger):
    # state = torch.load(filename)
    state = torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    if 'optimizer' in state and optimizer:
        optimizer.load_state_dict(state['optimizer'])
    logger.info('Model loaded : {}'.format(filename))


if __name__=='__main__':
    param = DistributedDataParallelKwargs(
            find_unused_parameters=False, check_reduction=False
    )
    accelerator = Accelerator(fp16=False, kwargs_handlers=[param])
    device = accelerator.device
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    src_tokenizer = spm.SentencePieceProcessor()
    src_tokenizer.load('wmt16_src.model')
    tgt_tokenizer = spm.SentencePieceProcessor()
    tgt_tokenizer.load('wmt16_tgt.model')

    # with open('restriced_vocab_src.json', 'r') as f:
    #    vocabs = json.load(f)
    
    # src_tokenizer.set_vocabulary(vocabs)
    
    # with open('restriced_vocab_tgt.json', 'r') as f:
    #    vocabs = json.load(f)
    
    # tgt_tokenizer.set_vocabulary(vocabs)

    ### config ###
    SRC_VOCAB_SIZE = src_tokenizer.get_piece_size()
    TGT_VOCAB_SIZE = tgt_tokenizer.get_piece_size()
    EMB_SIZE = 512
    NHEAD = 8
    FF_DIM = 1024
    BATCH_SIZE = 56 # 128 # 4
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    NUM_WORKERS = 8
    LR = 7e-4
    EPOCHS = 30
    ############### 

    dataset = WMT_Dataset('clean_wmt16_src_train.txt', 'clean_wmt16_tgt_train.txt', src_tokenizer, tgt_tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=WMT_collate, num_workers=NUM_WORKERS)
    valid_dataset = WMT_Dataset('wmt16_src_validation.txt', 'wmt16_tgt_validation.txt', src_tokenizer, tgt_tokenizer)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=WMT_collate, num_workers=NUM_WORKERS)

    model = Ptransformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, FF_DIM, 0.1, 0.1, src_tokenizer.pad_id())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    num_training_steps = EPOCHS * len(dataloader)
    
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=8000,
        num_training_steps=num_training_steps
    )
    criterion = nn.CrossEntropyLoss(ignore_index=src_tokenizer.pad_id())
    

    logger = get_logger(name='train',
                        file_path=os.path.join('.', 'train_log.log'),
                        stream=True)
    
#     load('model_021.pt', model, optimizer, logger)
    
    dataloader, valid_dataloader, model, optimizer = accelerator.prepare(
        dataloader, valid_dataloader, model, optimizer
    )
    train_begin = time.time()
    n_epoch = 0
    
    for epoch in range(0, EPOCHS):
        epoch_start_time = time.time()

        # train function
        train_loss  = train(model, optimizer, criterion, dataloader, src_tokenizer.pad_id(), train_begin, epoch, device, accelerator)
        logger.info('Epoch %d (Training) Loss %0.8f' % (epoch, train_loss))

        # evaluate function
        valid_loss, valid_BLEU = evaluate(model, criterion, valid_dataloader, src_tokenizer.pad_id(), tgt_tokenizer, device)
        logger.info('Epoch %d (Evaluate) Loss %0.8f BLEU %0.8f' % (epoch, valid_loss, valid_BLEU))
        
        # make_directory('checkpoint')
        save(os.path.join('checkpoint', f"model_{epoch:03d}.pt"), model, logger, accelerator)

        epoch_end_time = time.time()
        n_epoch += 1
        print(f'For {(epoch_end_time - epoch_start_time)/60:6.2f}, {n_epoch} Epoch Finished')
