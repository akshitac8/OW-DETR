#!/usr/bin/env bash

set -x

EXP_DIR=exps/OWDETR_t1
PY_ARGS=${@:1}

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19 --data_root '../data/OWDETR' --train_set 't1_train' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 45 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    ${PY_ARGS}

EXP_DIR=exps/OWDETR_t2
PY_ARGS=${@:1}

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 --data_root '../data/OWDETR' --train_set 't2_train' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 50 --lr 2e-5 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWDETR_t1/checkpoint0044.pth' \
    ${PY_ARGS}

EXP_DIR=exps/OWDETR_t2_ft
PY_ARGS=${@:1}

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 --data_root '../data/OWDETR' --train_set 't2_ft' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 100 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWDETR_t2/checkpoint0049.pth' \
    ${PY_ARGS}

EXP_DIR=exps/OWDETR_t3
PY_ARGS=${@:1}

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 --data_root '../data/OWDETR' --train_set 't3_train' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 106 --lr 2e-5 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWDETR_t2_ft/checkpoint0099.pth' \
    ${PY_ARGS}

EXP_DIR=exps/OWDETR_t3_ft
PY_ARGS=${@:1}

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 --data_root '../data/OWDETR' --train_set 't3_ft' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 161 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWDETR_t3/checkpoint0104.pth' \
    ${PY_ARGS}

EXP_DIR=exps/OWDETR_t4
PY_ARGS=${@:1}

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 --data_root '../data/OWDETR' --train_set 't4_train' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 171 --lr 2e-5 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWDETR_t3_ft/checkpoint0159.pth' \
    ${PY_ARGS}

EXP_DIR=exps/OWDETR_t4_ft
PY_ARGS=${@:1}

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 --data_root '../data/OWDETR' --train_set 't4_ft' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 302 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWDETR_t4/checkpoint0169.pth' \
    ${PY_ARGS}