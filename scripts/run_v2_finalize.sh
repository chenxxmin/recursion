#!/bin/bash
# 等所有 v2 训练进程自然结束（本轮存盘点）后：
#   1. 同步 logs/plots 到结果库并推送
#   2. 重建状态文档表格，代码库提交推送
set -u
cd "$(dirname "$0")/.."
ts() { date '+%F %T'; }

echo "[$(ts)] 等待全部 v2 训练进程结束..."
while pgrep -u "$(id -u)" -f 'core\.py config_tmp_action_v2' > /dev/null 2>&1; do
    sleep 120
done
sleep 30  # 等文件写完

echo "[$(ts)] 全部结束，同步结果库..."
for b in $(ls /data/cxm/recursion_v2/); do
    mkdir -p /data/cxm/recursion/action_v2/$b
    cp -r /data/cxm/recursion_v2/$b/logs /data/cxm/recursion_v2/$b/plots /data/cxm/recursion/action_v2/$b/ 2>/dev/null
done
cd /data/cxm/recursion
git add action_v2/
git commit -q -m "v2 final snapshot: all running experiments stopped at checkpoints $(date '+%m-%d %H:%M')" || echo "无新内容"
git pull --no-rebase --no-edit -q || true
git push origin HEAD

cd /home/cxm/recursion
python scripts/update_v2_status.py
git add reports/v2/ACTION_V2_STATUS.md scripts/update_v2_status.py
git commit -q -m "v2 status: final matrix after planned stop $(date '+%m-%d %H:%M')" || echo "无新内容"
git pull --no-rebase --no-edit -q || true
git push origin main

echo "[$(ts)] 收尾同步完成"
