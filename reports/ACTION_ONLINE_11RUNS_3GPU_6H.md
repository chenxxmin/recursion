# Action online 300k：11组、3卡、每组6h

用户已确认完整参数表，并再次明确每组定时6h。审阅版本：687505114f37bae8fc5c。
矩阵：MISS_LEN=1 的 l2/l4/l6；2 的 l2/l6；3 的 l2/l6；5 的 l2；8 的 l2/l4/l6。
共同参数：d1024 h4 r4，P127，N2 [(1,1),(2,3)]，TRAIN_LEN=OOD_LEN=64，
30万干净训练样本，每batch在线污损概率0.1；每epoch fresh test256。
AdamW LR3e-4/WD1/clip1；batch512/accum1；每轮重洗牌false；
FP32+TF32；seed17996；从头训练；cosine horizon6000。

定时写入 JSON 和 run.sh，均为 MAX_TRAIN_HOURS=6，每组独立从训练循环起计时。
每个epoch评估点检查，到时保存完整resume退出；排队/初始化/结束评估不计时。
正常早停仍生效，正常完成和早停也保存完整resume。
结束时评估1000条干净样本，汇总total/clean accuracy与曲线。

启动（目标机器仓库根目录）：
    bash scripts/run_action_online_300k_11runs_3gpu_6h.sh

指定物理候选卡池：
    bash scripts/run_action_online_300k_11runs_3gpu_6h.sh run --gpus 0,2,4,6 --background

脚本先比较空闲GPU数与预设3张，拿到全部运行锁后才启动。
不足3张则整批退出，无部分启动/降卡/自动等待。终端打印GPU型号、实际编号和任务分配。
第一批按配置顺序启动3组，余下8组FIFO接续先空出的卡。后台控制器继承运行锁。

默认输出：/data/cxm/final/<实验名>__<UTC时间>_<唯一编号>/，
内部 configs/logs/checkpoints/plots/metrics；控制器另建同根目录的批次目录。
启动时打印批次路径。再次执行创建新目录，不覆盖旧结果。

检查/停止/续跑（把占位符替换为启动时打印的批次路径）：
    bash scripts/run_action_online_300k_11runs_3gpu_6h.sh status --batch <批次路径>
    bash scripts/run_action_online_300k_11runs_3gpu_6h.sh stop --batch <批次路径>
    bash scripts/run_action_online_300k_11runs_3gpu_6h.sh resume --resume-batch <批次路径> --background

续跑对未完成且有完整存档的任务再分配每组6h，保留optimizer/scheduler/RNG与历史best；
不会自动无限续跑。正常完成的任务默认跳过，显式--only可选中。

可分发压缩包：bundles/action_online_300k_11runs_3gpu_6h_20260922.tar.gz
完整参数与冻结源码在同名目录；解压后也可直接 bash run.sh run --background。
只准备跨机交付；本批未在本机启动GPU训练。
