#!/bin/bash
# filepath: /mnt/8t/caizhuojun/gsam/Grounded-SAM-2/quick_push.sh

#!/bin/bash
# filepath: /8t/caizhuojun/gsam/Grounded-SAM-2/quick_push.sh

# 简单版本的 git 推送脚本

# 1. 设置 git 账户
echo "设置 git 账户..."
git config user.name "Glupayy"
git config user.email "lyy8ztc@outlook.com"  # 请替换为你的邮箱

# 2. 删除远程
echo "删除现有远程..."
git remote remove origin 2>/dev/null || echo "没有找到 origin 远程，跳过删除"

# 3. 添加远程
echo "添加新的远程..."
git remote add origin git@github.com:Glupayy/vggt_attn.git

# 获取提交信息
if [ $# -eq 0 ]; then
    commit_message="Auto commit $(date '+%Y-%m-%d %H:%M:%S')"
else
    commit_message="$*"
fi

# export http_proxy=http://10.253.34.172:6666
# export https_proxy=http://10.253.34.172:6666
echo "提交信息: $commit_message"
# git config http.proxy http://10.253.34.172:6666
# git config https.proxy http://10.253.34.172:6666

# 执行 git 操作
git add .
git commit -m "$commit_message"
git branch -M master
git push origin HEAD:master