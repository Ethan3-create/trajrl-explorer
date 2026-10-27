#!/bin/bash
# ============================================================
# TrajRL Explorer - 阿里云 ECS 一键安装脚本
# ============================================================
# 运行前准备:
#   1. 购买阿里云 ECS (建议 Ubuntu 22.04, 2核4G以上)
#   2. 开放安全组: 入方向允许 TCP 80, 443, 8601
#   3. SSH 登录服务器: ssh root@your-ecs-ip
#   4. 上传本脚本: scp install.sh root@your-ecs-ip:/root/
#   5. 执行: bash install.sh
# ============================================================

set -e

# 配置
APP_NAME="trajrl-explorer"
APP_DIR="/opt/${APP_NAME}"
REPO_URL="https://github.com/your-username/trajrl-explorer.git"  # 修改为你的仓库地址
PORT=${TRAJRL_PORT:-8601}
WORKERS=${TRAJRL_WORKERS:-4}
SERVICE_USER="www-data"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== TrajRL Explorer 阿里云部署脚本 ===${NC}"
echo ""

# 1. 系统更新
echo -e "${YELLOW}[1/8] 更新系统包...${NC}"
apt-get update -qq
apt-get upgrade -y -qq

# 2. 安装依赖
echo -e "${YELLOW}[2/8] 安装系统依赖...${NC}"
apt-get install -y -qq \
    python3 \
    python3-pip \
    python3-venv \
    nginx \
    git \
    curl \
    wget \
    ufw \
    htop

# 3. 配置防火墙
echo -e "${YELLOW}[3/8] 配置防火墙...${NC}"
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 8601/tcp
ufw --force enable

# 4. 创建应用目录
echo -e "${YELLOW}[4/8] 创建应用目录...${NC}"
mkdir -p ${APP_DIR}
useradd -r -s /bin/false ${SERVICE_USER} 2>/dev/null || true

# 5. 克隆仓库或从本地复制
echo -e "${YELLOW}[5/8] 部署应用代码...${NC}"
if [ -d "/root/trajrl-explorer" ]; then
    # 从本地目录复制（如果你手动上传了项目文件夹）
    cp -r /root/trajrl-explorer/* ${APP_DIR}/
    echo "从 /root/trajrl-explorer 复制代码"
elif [ -n "$REPO_URL" ] && [ "$REPO_URL" != "https://github.com/your-username/trajrl-explorer.git" ]; then
    # 从 GitHub 克隆
    git clone ${REPO_URL} ${APP_DIR}
    echo "从 ${REPO_URL} 克隆代码"
else
    echo -e "${RED}错误: 请先将项目文件夹上传到 /root/trajrl-explorer，或修改 REPO_URL${NC}"
    exit 1
fi

# 设置权限
chown -R ${SERVICE_USER}:${SERVICE_USER} ${APP_DIR}
chmod -R 755 ${APP_DIR}

# 6. 安装 Python 依赖
echo -e "${YELLOW}[6/8] 安装 Python 依赖...${NC}"
cd ${APP_DIR}
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "依赖安装完成"

# 7. 配置 Nginx 反向代理
echo -e "${YELLOW}[7/8] 配置 Nginx...${NC}"
cat > /etc/nginx/sites-available/${APP_NAME} << 'EOF'
server {
    listen 80;
    server_name _;  # 接受任意域名或 IP

    # 静态文件（可选：如果希望 Nginx 直接处理静态文件以提升性能）
    # location /css/ {
    #     alias /opt/trajrl-explorer/web_demo/css/;
    #     expires 1d;
    # }
    # location /js/ {
    #     alias /opt/trajrl-explorer/web_demo/js/;
    #     expires 1d;
    # }

    # 所有请求代理到 Gunicorn
    location / {
        proxy_pass http://127.0.0.1:8601;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
EOF

# 启用站点
ln -sf /etc/nginx/sites-available/${APP_NAME} /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default 2>/dev/null || true
nginx -t
systemctl restart nginx
systemctl enable nginx

# 8. 配置 systemd 服务
echo -e "${YELLOW}[8/8] 配置系统服务...${NC}"
cat > /etc/systemd/system/${APP_NAME}.service << EOF
[Unit]
Description=TrajRL Explorer Web Demo
After=network.target nginx.service

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_USER}
WorkingDirectory=${APP_DIR}
Environment="PATH=${APP_DIR}/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONUNBUFFERED=1"
Environment="TRAJRL_PORT=${PORT}"
ExecStart=${APP_DIR}/venv/bin/python ${APP_DIR}/web_server.py --host 127.0.0.1 --port ${PORT}
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ${APP_NAME}

# 启动服务
echo -e "${GREEN}启动 TrajRL Explorer...${NC}"
systemctl start ${APP_NAME}

# 等待服务启动
sleep 3

# 检查状态
if systemctl is-active --quiet ${APP_NAME}; then
    echo -e "${GREEN}✅ 服务启动成功!${NC}"
    echo ""
    echo -e "${GREEN}=== 访问地址 ===${NC}"
    echo "HTTP:  http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || hostname -I | awk '{print $1}'):80"
    echo "(阿里云安全组需开放 80 端口)"
    echo ""
    echo -e "${GREEN}=== 管理命令 ===${NC}"
    echo "查看状态:  systemctl status ${APP_NAME}"
    echo "查看日志:  journalctl -u ${APP_NAME} -f"
    echo "重启服务:  systemctl restart ${APP_NAME}"
    echo "停止服务:  systemctl stop ${APP_NAME}"
    echo ""
    echo -e "${YELLOW}提示: 默认通过 Nginx 80 端口访问。如需 HTTPS/域名，请配置 SSL 证书。${NC}"
else
    echo -e "${RED}❌ 服务启动失败，查看日志:${NC}"
    journalctl -u ${APP_NAME} --no-pager -n 20
    exit 1
fi
