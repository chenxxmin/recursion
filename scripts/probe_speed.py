"""探针：定位 action d1024l4 训练慢的原因并量化加速手段。

测量项（同一配置、同一 GPU，互不修改仓库代码）：
  1. 纯数据迭代耗时（CPU 数据管线是否瓶颈）
  2. train_epoch 耗时（fp32 基线）
  3. evaluate 耗时（摊薄到每 epoch）
  4. bf16 autocast 下的 train_epoch / evaluate
  5. torch.compile 后的 train_epoch（含预热）

用法: CUDA_VISIBLE_DEVICES=<gpu> python probe_speed.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import torch

import experiment
from training import train_epoch, evaluate


def build_ctx():
    base = json.load(open('src/config.json'))
    exp = next(e for e in json.load(open('experiments/action_misslen2_n2_scaleup.json'))['experiments']
               if e['name'] == 'action_d1024l4r4h4_P127_tr64ood128_N2_randmiss0.1len2_seed17996')
    merged_main = dict(base['main'])
    merged_main.update(base.get('action', {}))
    merged_main.update(exp['config'])
    merged_main['SAVE_PATH'] = '/tmp/probe_speed.pth'
    cfg = dict(base)
    cfg['main'] = merged_main
    cfg['_BATCH_RUN_MERGED'] = True
    return experiment._prepare_action(cfg), merged_main


def time_fn(fn, n=1, sync=True):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def main():
    ctx, cfg = build_ctx()
    device = 'cuda'
    model = ctx['model'].to(device)
    train_loader = ctx['train_loader']
    test_loader = ctx['test_loader']
    num_mask = ctx['num_mask']
    extra_kwargs_fn = ctx['extra_kwargs_fn']
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['LR'],
                                  weight_decay=cfg['WEIGHT_DECAY'])
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M, "
          f"train batches/epoch: {len(train_loader)}, device: {device}")

    def run_train_epoch(m=None):
        return train_epoch(m or model, train_loader, optimizer, device,
                           num_mask=num_mask, extra_kwargs_fn=extra_kwargs_fn)

    def run_eval(m=None):
        return evaluate(m or model, test_loader, device,
                        num_mask=num_mask, extra_kwargs_fn=extra_kwargs_fn)

    # 0. 纯数据迭代（CPU 管线成本）
    t_data = time_fn(lambda: [b for b in train_loader])
    print(f"[data]  full loader iteration: {t_data:.2f}s/epoch")

    # 1. fp32 基线
    t_train = time_fn(run_train_epoch, n=3)
    t_eval = time_fn(run_eval)
    print(f"[fp32]  train: {t_train:.2f}s/epoch | eval: {t_eval:.2f}s "
          f"(amortized {t_eval/20:.2f}s/epoch @EVAL_INTERVAL=20)")

    # 2. TF32（不改数值路径，只放开张量核 fp32 加速档）
    torch.backends.cuda.matmul.allow_tf32 = True
    run_train_epoch()  # warmup
    t_train_tf32 = time_fn(run_train_epoch, n=3)
    t_eval_tf32 = time_fn(run_eval)
    print(f"[tf32]  train: {t_train_tf32:.2f}s/epoch ({t_train/t_train_tf32:.2f}x) | "
          f"eval: {t_eval_tf32:.2f}s")
    torch.backends.cuda.matmul.allow_tf32 = False

    # 3. bf16 autocast（包整 epoch；backward 在 autocast 外也安全）
    def run_train_amp():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return run_train_epoch()

    def run_eval_amp():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return run_eval()

    run_train_amp()  # warmup
    t_train_amp = time_fn(run_train_amp, n=3)
    t_eval_amp = time_fn(run_eval_amp)
    print(f"[bf16]  train: {t_train_amp:.2f}s/epoch ({t_train/t_train_amp:.2f}x) | "
          f"eval: {t_eval_amp:.2f}s ({t_eval/max(t_eval_amp,1e-9):.2f}x)")

    # 4. TF32 + torch.compile
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        model_c = torch.compile(model)
        t0 = time.time()
        run_train_epoch(model_c)  # compile warmup
        torch.cuda.synchronize()
        t_warm = time.time() - t0
        t_train_c = time_fn(lambda: run_train_epoch(model_c), n=3)
        print(f"[comp]  train: {t_train_c:.2f}s/epoch ({t_train/t_train_c:.2f}x) "
              f"| compile warmup: {t_warm:.0f}s")
    except Exception as e:
        print(f"[comp]  failed: {e}")

    print(f"[mem]   peak: {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")


if __name__ == '__main__':
    main()
