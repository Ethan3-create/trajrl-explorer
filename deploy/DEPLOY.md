# TrajRL Explorer - 阿里云 ECS 部署指南

## 前置条件

| 项目 | 要求 |
|---|---|
| 阿里云账号 | 已实名认证 |
| ECS 实例 | 推荐 Ubuntu 22.04 LTS，2核4G 以上 |
| 公网 IP | 分配公网 IPv4 地址 |
| 安全组 | 入方向开放 TCP 80（HTTP）、443（HTTPS，可选）、22（SSH） |

## 部署流程（3 步走）

### 第 1 步：准备 GitHub 仓库

1. 在 GitHub 创建新仓库（如 `trajrl-explorer`）
2. 将本地项目 push 到仓库：

```bash
# 在项目根目录执行
git init
git add .
git commit -m "Initial commit: TrajRL Explorer interactive demo"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/trajrl-explorer.git
git push -u origin main
```

3. 修改 `deploy/install.sh` 中的 `REPO_URL` 为你的仓库地址：
```bash
REPO_URL="https://github.com/YOUR_USERNAME/trajrl-explorer.git"
```

### 第 2 步：购买并配置阿里云 ECS

1. 登录 [阿里云控制台](https://ecs.console.aliyun.com)
2. 创建 ECS 实例：
   - **地域**：选择离你目标用户近的地域（如华东 1 杭州）
   - **镜像**：Ubuntu 22.04 LTS 64位
   - **实例规格**：2核(vCPU) 4GiB 内存（ecs.t6-c1m2.large，约 ¥0.15/小时）
   - **带宽**：按量付费，峰值 5Mbps（¥0.8/GB）
   - **安全组**：添加规则——
     - 允许 SSH (TCP 22) 源 0.0.0.0/0
     - 允许 HTTP (TCP 80) 源 0.0.0.0/0
     - 允许 HTTPS (TCP 443) 源 0.0.0.0/0（可选）

3. 记录公网 IP 地址（如 `47.96.XX.XX`）

### 第 3 步：执行一键部署

**方式 A：通过 GitHub 自动克隆（推荐）**

```bash
# 本地终端执行
scp deploy/install.sh root@47.96.XX.XX:/root/
ssh root@47.96.XX.XX
bash /root/install.sh
```

**方式 B：手动上传项目文件夹（网络受限时）**

```bash
# 本地打包
zip -r trajrl-explorer.zip trajrl-explorer/

# 上传到服务器
scp trajrl-explorer.zip root@47.96.XX.XX:/root/

# 登录服务器并解压
ssh root@47.96.XX.XX
unzip /root/trajrl-explorer.zip -d /root/

# 执行安装脚本
bash /root/trajrl-explorer/deploy/install.sh
```

## 部署后验证

1. **查看服务状态**：
```bash
systemctl status trajrl-explorer
```

2. **查看日志**：
```bash
journalctl -u trajrl-explorer -f
```

3. **浏览器访问**：
```
http://47.96.XX.XX
```

应该看到 TrajRL Explorer 首页，统计数字正常加载。

## 运维管理

| 操作 | 命令 |
|---|---|
| 重启服务 | `systemctl restart trajrl-explorer` |
| 停止服务 | `systemctl stop trajrl-explorer` |
| 查看日志 | `journalctl -u trajrl-explorer -f` |
| 查看资源 | `htop` |
| 更新代码 | `cd /opt/trajrl-explorer && git pull && systemctl restart trajrl-explorer` |

## 进阶配置

### 配置域名 + HTTPS

1. 购买域名并添加 A 记录指向 ECS 公网 IP
2. 安装 certbot：
```bash
apt-get install certbot python3-certbot-nginx
certbot --nginx -d your-domain.com
```
3. certbot 会自动配置 Nginx SSL 并设置自动续期

**注意**：中国大陆域名需完成 ICP 备案才能使用 80/443 端口。

### 使用 Gunicorn 替代内置 HTTP 服务器

编辑 `/etc/systemd/system/trajrl-explorer.service`：

```ini
[Service]
ExecStart=/opt/trajrl-explorer/venv/bin/gunicorn \
    -w 4 \
    -b 127.0.0.1:8601 \
    --timeout 120 \
    --access-logfile /var/log/trajrl/access.log \
    --error-logfile /var/log/trajrl/error.log \
    wsgi:app
```

然后重启：
```bash
mkdir -p /var/log/trajrl
systemctl daemon-reload
systemctl restart trajrl-explorer
```

### 数据文件独立挂载（大数据集时）

如果数据集超过 100MB，不适合放在 GitHub 仓库：

1. 上传到阿里云 OSS 或 NAS
2. 修改 `web_server.py` 中的 `SAMPLE_PATH`：
```python
SAMPLE_PATH = os.environ.get("TRAJRL_DATA_PATH", 
    os.path.join(SCRIPT_DIR, "outputs", "porto_sample1w", "sample.jsonl"))
```
3. 在服务器上挂载数据目录：
```bash
# 例如从 OSS 下载
wget -O /data/porto_sample1w/sample.jsonl https://your-oss-bucket.oss-cn-hangzhou.aliyuncs.com/sample.jsonl
export TRAJRL_DATA_PATH=/data/porto_sample1w/sample.jsonl
```

## 故障排查

| 问题 | 排查步骤 |
|---|---|
| 浏览器无法访问 | 检查安全组是否开放 80；检查 `ufw status` |
| 502 Bad Gateway | 检查 `systemctl status trajrl-explorer`；检查 `nginx -t` |
| 页面空白 | 检查浏览器控制台是否有 CDN 资源加载失败 |
| 地图不显示 | 检查服务器能否访问 `tile.openstreetmap.de`：`curl -I https://tile.openstreetmap.de/0/0/0.png` |
| 数据加载慢 | 检查内存使用：`free -h`；考虑升级 ECS 配置 |
| 服务频繁重启 | 查看日志：`journalctl -u trajrl-explorer -n 50` |

## 费用估算（按量付费）

| 配置 | 单价 | 月费用（每天运行 8 小时） |
|---|---|---|
| 2核4G (ecs.t6-c1m2.large) | ¥0.148/小时 | ~¥35 |
| 5Mbps 带宽（按量） | ¥0.8/GB | 视流量，通常 <¥10 |
| 40GB 高效云盘 | ¥0.0144/GB/月 | ~¥0.6 |
| **合计** | — | **~¥45/月** |

**省钱技巧**：
- 不用时停止 ECS 实例（仅保留云盘，¥0.0144/GB/月）
- 使用抢占式实例（Spot），价格约 1/3，但可能被回收

## 联系与支持

如有部署问题，请提交 GitHub Issue 或查看项目 README。
