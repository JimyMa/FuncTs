import argparse
from dataclasses import dataclass

import functs

import numpy as np

import torch
from functs.benchmark.ai_model import fcos, lstm, nasrnn, seq2seq, ssd, yolact, yolov3
from functs.benchmark.simpleops import attention
from functs.benchmark.utils import process_feat_batch
from functs.utils import evaluate_func, evaluate_task

torch._dynamo.reset()


parser = argparse.ArgumentParser()
parser.add_argument("--bs", default=1, type=int)
parser.add_argument("--platform", default="1660ti", type=str)
parser.add_argument("--maxlength", default=50, type=int)

arguments = parser.parse_args()

with torch.no_grad():
    bs = arguments.bs

    # cv models
    cv_model = [
        yolov3.yolov3_bbox.YOLOV3BBox().cuda().eval(),
        ssd.ssd_bbox.SSDBBox().cuda().eval(),
        yolact.yolact_mask.YolactBBoxMask().cuda().eval(),
        fcos.fcos_bbox.FCOSBBox().cuda().eval(),
    ]

    cv_model_name = [model.__class__.__name__ for model in cv_model]

    feats = [
        process_feat_batch(yolov3.feats, bs),
        process_feat_batch(ssd.feats, bs),
        process_feat_batch(yolact.feats, bs),
        process_feat_batch(fcos.feats, bs),
    ]

    example_in = [feats[0] for feats in feats]
    cv_jit_model = [torch.jit.freeze(torch.jit.script(model)) for model in cv_model]
    cv_dynamo_model = [torch.compile(model, dynamic=True) for model in cv_model]
    cv_nvfuser_model = [torch.jit.freeze(torch.jit.script(model)) for model in cv_model]
    cv_functs_model = [
        functs.jit.build(functs.jit.script(model, backend="aot"), example_in)
        for model, example_in in zip(cv_model, example_in)
    ]

    backbone_time = [1.54, 1.35, 3.41, 5.136]

    def task(fn, feats):
        return lambda idx: fn(*feats[idx % len(feats)])

    eager_mode_latency = np.array(
        [
            float(
                evaluate_task(
                    task(model, feats),
                    name="{}_eager".format(cv_model_name),
                    run_duration=3.0,
                ).avg(round_to="ms")
            )
            + backbone_time
            for model, cv_model_name, feats, backbone_time in zip(
                cv_model, cv_model_name, feats, backbone_time
            )
        ]
    )
    jit_mode_latency = np.array(
        [
            float(
                evaluate_task(
                    task(model, feats),
                    name="{}_jit".format(cv_model_name),
                    run_duration=3.0,
                ).avg(round_to="ms")
            )
            + backbone_time
            for model, cv_model_name, feats, backbone_time in zip(
                cv_jit_model, cv_model_name, feats, backbone_time
            )
        ]
    )
    functs_mode_latency = np.array(
        [
            float(
                evaluate_task(
                    task(model, feats),
                    name="{}_functs".format(cv_model_name),
                    run_duration=3.0,
                ).avg(round_to="ms")
            )
            + backbone_time
            for model, cv_model_name, feats, backbone_time in zip(
                cv_functs_model, cv_model_name, feats, backbone_time
            )
        ]
    )
    dynamo_mode_latency = np.array(
        [
            float(
                evaluate_task(
                    task(model, feats),
                    name="{}_dynamo".format(cv_model_name),
                    run_duration=3.0,
                ).avg(round_to="ms")
            )
            + backbone_time
            for model, cv_model_name, feats, backbone_time in zip(
                cv_dynamo_model, cv_model_name, feats, backbone_time
            )
        ]
    )

    torch._C._jit_set_nvfuser_enabled(True)
    nvfuser_mode_latency = np.array(
        [
            float(
                evaluate_task(
                    task(model, feats),
                    name="{}_nvfuser".format(cv_model_name),
                    run_duration=3.0,
                ).avg(round_to="ms")
            )
            + backbone_time
            for model, cv_model_name, feats, backbone_time in zip(
                cv_nvfuser_model, cv_model_name, feats, backbone_time
            )
        ]
    )
    torch._C._jit_set_nvfuser_enabled(False)

    cv_latency = np.stack(
        [
            eager_mode_latency,
            dynamo_mode_latency,
            nvfuser_mode_latency,
            jit_mode_latency,
            functs_mode_latency,
        ]
    )
    print(cv_latency)

    BATCH_SIZE = arguments.bs
    INPUT_SIZE = 256
    HIDDEN_SIZE = 256
    SEQ_LEN = arguments.maxlength

    LSTM_NUM_LAYERS = 10

    SEQ2SEQ_OUTPUT_SIZE = 500
    SEQ2SEQ_MAX_LENGTH = SEQ_LEN

    ATTENTION_NUM_HEAD = 12
    ATTENTION_START_LEN = 32
    ATTENTION_SIZE_PER_HEAD = 64

    seq2seq_model = (
        seq2seq.seq2seq.AttnDecoderRNN(HIDDEN_SIZE, SEQ2SEQ_OUTPUT_SIZE, dropout_p=0.1)
        .cuda()
        .eval()
    )

    nlp_model = [
        nasrnn.nasrnn.NasRNN(BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE).cuda().eval(),
        lstm.lstm.LSTM(INPUT_SIZE, HIDDEN_SIZE, LSTM_NUM_LAYERS).cuda().eval(),
        seq2seq_model,
        attention.attention.Attention(
            ATTENTION_NUM_HEAD, ATTENTION_SIZE_PER_HEAD, SEQ_LEN
        )
        .cuda()
        .eval(),
    ]

    nlp_model_name = [model.__class__.__name__ for model in nlp_model]

    def generate_seq2seq_input():
        def gen_mask_from_sequence(std):
            bs = std.shape[0]
            padded_std = torch.zeros((bs, SEQ2SEQ_MAX_LENGTH), dtype=std.dtype).cuda()
            padded_std[:, : std.shape[1]] = std
            mask = torch.zeros(bs, SEQ2SEQ_MAX_LENGTH, SEQ2SEQ_OUTPUT_SIZE).cuda()
            mask[
                torch.arange(bs).unsqueeze(1),
                torch.arange(SEQ2SEQ_MAX_LENGTH).unsqueeze(0),
                padded_std,
            ] = 1000000.0
            mask = mask.transpose(0, 1).contiguous().clone()
            return mask

        seq2seq_h = torch.randn(BATCH_SIZE, HIDDEN_SIZE).cuda()
        seq2seq_c = torch.randn(BATCH_SIZE, HIDDEN_SIZE).cuda()
        sos = torch.full(
            (BATCH_SIZE,), seq2seq_model.SOS_token, dtype=torch.int64, device="cuda"
        )
        std = []
        for i in range(BATCH_SIZE):
            l = 10
            lst = list(range(1, l))
            lst.append(0)
            assert len(lst) <= SEQ2SEQ_MAX_LENGTH
            # pad to MAX_LENGTH
            lst = lst + [0] * (SEQ2SEQ_MAX_LENGTH - len(lst))
            std.append(lst)
        std = torch.tensor(std).cuda()
        seq2seq_mask = gen_mask_from_sequence(std)
        seq2seq_encoder_output = torch.randn(
            SEQ2SEQ_MAX_LENGTH, BATCH_SIZE, HIDDEN_SIZE
        ).cuda()
        return [seq2seq_encoder_output, seq2seq_mask, seq2seq_h, seq2seq_c]

    def generate_attention_input():
        x = torch.randn(
            BATCH_SIZE, ATTENTION_NUM_HEAD, 1, ATTENTION_SIZE_PER_HEAD
        ).cuda()
        k = torch.zeros(
            BATCH_SIZE,
            ATTENTION_NUM_HEAD,
            SEQ_LEN,
            ATTENTION_SIZE_PER_HEAD,
            dtype=torch.float32,
            device="cuda",
        )
        k[:, :, :ATTENTION_START_LEN, :] = torch.randn(
            BATCH_SIZE,
            ATTENTION_NUM_HEAD,
            ATTENTION_START_LEN,
            ATTENTION_SIZE_PER_HEAD,
            dtype=torch.float32,
            device="cuda",
        )
        v = torch.zeros(
            BATCH_SIZE,
            ATTENTION_NUM_HEAD,
            SEQ_LEN,
            ATTENTION_SIZE_PER_HEAD,
            dtype=torch.float32,
            device="cuda",
        )
        v[:, :, :ATTENTION_START_LEN, :] = torch.randn(
            BATCH_SIZE,
            ATTENTION_NUM_HEAD,
            ATTENTION_START_LEN,
            ATTENTION_SIZE_PER_HEAD,
            dtype=torch.float32,
            device="cuda",
        )
        return [x, k, v]

    inputs = [
        [torch.rand([SEQ_LEN, BATCH_SIZE, INPUT_SIZE]).cuda().float()],
        [torch.randn([SEQ_LEN, BATCH_SIZE, INPUT_SIZE]).cuda().float()],
        generate_seq2seq_input(),
        generate_attention_input(),
    ]

    nlp_jit_model = [torch.jit.freeze(torch.jit.script(model)) for model in nlp_model]
    nlp_dynamo_model = [torch.compile(model, dynamic=True) for model in nlp_model]
    nlp_nvfuser_model = [
        torch.jit.freeze(torch.jit.script(model)) for model in nlp_model
    ]
    nlp_functs_model = [functs.jit.script(model) for model in nlp_model]

    eager_mode_latency = np.array(
        [
            float(
                evaluate_func(
                    model,
                    inputs,
                    name="{}_eager".format(nlp_model_name),
                    run_duration=3.0,
                ).avg(round_to="ms")
            )
            for model, nlp_model_name, inputs in zip(nlp_model, nlp_model_name, inputs)
        ]
    )
    jit_mode_latency = np.array(
        [
            float(
                evaluate_func(
                    model,
                    inputs,
                    name="{}_jit".format(nlp_model_name),
                    run_duration=3.0,
                ).avg(round_to="ms")
            )
            for model, nlp_model_name, inputs in zip(
                nlp_jit_model, nlp_model_name, inputs
            )
        ]
    )
    functs_mode_latency = np.array(
        [
            float(
                evaluate_func(
                    model,
                    inputs,
                    name="{}_functs".format(nlp_model_name),
                    run_duration=3.0,
                ).avg(round_to="ms")
            )
            for model, nlp_model_name, inputs in zip(
                nlp_functs_model, nlp_model_name, inputs
            )
        ]
    )
    dynamo_mode_latency = np.array(
        [
            float(
                evaluate_func(
                    model,
                    inputs,
                    name="{}_dynamo".format(nlp_model_name),
                    run_duration=3.0,
                ).avg(round_to="ms")
            )
            for model, nlp_model_name, inputs in zip(
                nlp_dynamo_model, nlp_model_name, inputs
            )
        ]
    )
    torch._C._jit_set_nvfuser_enabled(True)
    nvfuser_mode_latency = np.array(
        [
            float(
                evaluate_func(
                    model,
                    inputs,
                    name="{}_nvfuser".format(nlp_model_name),
                    run_duration=3.0,
                ).avg(round_to="ms")
            )
            for model, nlp_model_name, inputs in zip(
                nlp_nvfuser_model, nlp_model_name, inputs
            )
        ]
    )
    torch._C._jit_set_nvfuser_enabled(False)

    nlp_latency = np.stack(
        [
            eager_mode_latency,
            dynamo_mode_latency,
            nvfuser_mode_latency,
            jit_mode_latency,
            functs_mode_latency,
        ]
    )
    print(nlp_latency)

    latency = np.concatenate([cv_latency, nlp_latency], axis=-1)

    torch.save(latency, "latency_{}.pt".format(arguments.platform))
