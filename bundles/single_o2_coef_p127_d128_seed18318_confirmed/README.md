# 已确认实验运行包

Linux；Python 3.10+；预先安装匹配目标机器 CUDA 的 PyTorch 及 requirements.txt 中的依赖。
本包包含当前实际代码快照，不依赖目标机器旧仓库。

检查：bash run.sh check

启动：bash run.sh run --gpus 0,1 --output /your/output --background

分发到多机时，用 --only experiment_name[,experiment_name] 选择各机负责的实验。

空闲排队：bash run.sh run --gpus 2,3,0,1 --output /your/output --wait-for-idle --poll-seconds 300 --background

排队时GPU列表是候选池；一张空闲即可先启动一条，等待不计入训练预算。

进度：bash run.sh status --output /your/output

存档停止：bash run.sh stop --output /your/output

EXPERIMENT_PYTHON 环境变量可指定 Python。输出目录必须全新，避免覆盖旧结果。
每条定时见 run.sh 中 TIMERS_JSON 和 PARAMETERS.md；从进入训练循环开始独立计时，到下一个 eval 点完整存盘，退出码 42 为正常定时/手动存档退出。无定时为 null。12h 是训练上限，不含排队和初始化；早停条件仍生效。
参数修改后应重新审阅打包；运行脚本只允许选择 GPU、输出位置和实验子集。
在线 train 指标不是固定权重全训练集评估；本包不自动添加额外实验或全量复评。
