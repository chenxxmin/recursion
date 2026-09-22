# Action online 实验运行包

11 组；3 卡并行；每组独立定时 6h。
完整参数见 PARAMETERS.md；JSON、脚本和 manifest 的定时必须一致。

环境：Linux、Python 3.10+、匹配机器 CUDA 的 PyTorch、numpy、matplotlib。
可用 EXPERIMENT_PYTHON 指定解释器。训练代码已随包冻结，不依赖机器上的旧版本。

```bash
bash run.sh check
# 自动从可见且获准的 GPU 中选择 3 张空闲卡，后台运行
bash run.sh run --background
# 也可指定物理 GPU 候选池；仍必须至少有 3 张空闲卡
bash run.sh run --gpus 0,2,4,6 --background
```

启动检查：显存 <100 MiB、利用率=0、没有 compute 进程且能取得运行锁。
先比较空闲数与预设卡数，数量不足则整批退出，不启动任何训练，也不自动等待。
检查和取得全部 GPU 锁在后台分离前完成；终端打印实际卡号、型号和首批任务表。
后续任务按 JSON 顺序接续空出的卡，训练失败不自动重试。
继承 CUDA_VISIBLE_DEVICES 限制；在打包本机上还严格限制为 GPU 0–3。

默认根目录：/data/cxm/final（可用 --output-root 指定其他机器上的路径）。
每个实验独立建立 <实验名>__<UTC时间>_<唯一编号>/，内部按 configs、logs、
checkpoints、plots、metrics 分类。批次控制目录也在同一根目录内，
包含 configs、logs、state；启动时打印其绝对路径。再次运行创建新目录，不覆盖旧结果。

```bash
bash run.sh status --batch /data/cxm/final/<启动时打印的批次目录>
bash run.sh stop --batch /data/cxm/final/<启动时打印的批次目录>
# 只续跑有完整存档、尚未正常完成的任务；每条再计时 6h
bash run.sh resume --resume-batch /data/cxm/final/<旧批次目录> --background
# 显式选择正常完成的任务也可续跑，仍要求预设的 3 张空闲卡
bash run.sh resume --resume-batch /data/cxm/final/<旧批次目录> --only <完整实验名> --background
```

6h 从每条实验进入训练循环开始，在每个 epoch 的 eval 点检查；排队和初始化不计时，
可能超出一个 epoch 及存档时间。EPOCHS=6000 始终是 cosine horizon。
既有早停仍生效：最近10次新鲜测试准确率均值 >99%，或3000 epoch 无提升。
正常完成、早停、定时和手动存档退出都保留完整 model_resume.pth；
每个常规 eval 另写 model_latest.pth，最佳权重为 model_best.pth。
resume 校验配置、完整状态、base LR、WD 和 scheduler horizon，再从下一 epoch 继续。
续跑创建新的 run 目录，原文件保留；不会自动重置 scheduler 或无限续跑。

训练结束后在同一卡上用 1000 条干净样本、seed=123、
TRAIN_LEN 长度评估 best 权重（缺失时用完整最新存档）；汇总日志中的 total accuracy。
写 metrics/summary.json、metrics/learning_curve.csv、plots/learning_curve.png。
clean 评估和曲线生成不计入 6h 训练预算。手动停止可跳过评估以优先保存退出。

本包不自动安装依赖，不会回退 CPU 或降低预设并行卡数。
