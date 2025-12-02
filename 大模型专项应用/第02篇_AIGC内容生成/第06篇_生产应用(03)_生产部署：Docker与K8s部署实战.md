# 第35章 生产环境部署

> **难度**: ⭐⭐⭐⭐⭐ | **推荐度**: ⭐⭐⭐⭐⭐

## 35.1 Docker化部署

### 35.1.1 SDXL API服务

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 安装Python
RUN apt-get update && apt-get install -y python3-pip

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . /app
WORKDIR /app

# 下载模型
RUN python download_models.py

# 启动服务
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# 构建
docker build -t sdxl-api .

# 运行
docker run --gpus all -p 8000:8000 sdxl-api
```

---

## 35.2 Kubernetes编排

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aigc-workers
spec:
  replicas: 4
  selector:
    matchLabels:
      app: aigc-worker
  template:
    metadata:
      labels:
        app: aigc-worker
    spec:
      containers:
      - name: worker
        image: sdxl-api:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: WORKER_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
```

---

## 35.3 负载均衡

```python
# Nginx配置
upstream aigc_backend {
    least_conn;  # 最少连接
    server 192.168.1.101:8000 weight=1 max_fails=3;
    server 192.168.1.102:8000 weight=1 max_fails=3;
    server 192.168.1.103:8000 weight=1 max_fails=3;
    server 192.168.1.104:8000 weight=1 max_fails=3;
}

server {
    listen 80;

    location /api/generate {
        proxy_pass http://aigc_backend;
        proxy_timeout 300s;  # 生成可能耗时
        proxy_read_timeout 300s;
    }
}
```

---

## 35.4 监控告警

```python
# Prometheus指标暴露
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# 定义指标
request_count = Counter('aigc_requests_total', 'Total requests', ['endpoint', 'status'])
generation_duration = Histogram('aigc_generation_seconds', 'Generation time')
gpu_memory = Gauge('aigc_gpu_memory_used_bytes', 'GPU memory', ['gpu_id'])
queue_size = Gauge('aigc_queue_size', 'Task queue size')

# 在生成函数中埋点
@app.post("/generate")
async def generate(request):
    with generation_duration.time():
        try:
            result = await generate_image(request)
            request_count.labels(endpoint='generate', status='success').inc()
            return result
        except Exception as e:
            request_count.labels(endpoint='generate', status='failed').inc()
            raise

# 启动metrics server
start_http_server(9090)
```

---

## 35.5 高可用架构

```
┌─────────────┐
│   Cloudflare │  ← CDN/DDoS防护
└──────┬───────┘
       │
┌──────▼───────┐
│ Load Balancer│  ← Nginx/HAProxy
└──────┬───────┘
       │
┌──────┴───────────────────┐
│                          │
▼                          ▼
API Server 1           API Server 2
(GPU 0-1)              (GPU 2-3)
│                          │
└──────┬───────────────────┘
       │
┌──────▼───────┐
│  Redis Queue │  ← 任务队列
└──────────────┘
       │
┌──────▼───────┐
│  PostgreSQL  │  ← 任务记录
└──────────────┘
```

---

## 35.6 备份与灾难恢复

```bash
# 模型备份
rsync -avz /models/ backup_server:/backup/models/

# 数据库备份
pg_dump aigc_db | gzip > backup_$(date +%Y%m%d).sql.gz

# S3图片备份
aws s3 sync /output/ s3://aigc-backup/outputs/

# 定时任务
0 2 * * * /scripts/backup.sh  # 每天凌晨2点备份
```

---

## 35.7 安全加固

```python
# API认证
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != os.getenv("API_SECRET"):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

@app.post("/generate")
async def generate(request, token = Depends(verify_token)):
    # 已认证,处理请求
    ...

# Rate Limiting
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/generate")
@limiter.limit("10/minute")  # 每IP限10次/分钟
async def generate(request):
    ...
```

---

## 35.8 总结

**生产部署关键点**:
1. Docker化 → 可移植
2. K8s编排 → 自动扩缩容
3. 负载均衡 → 高可用
4. 监控告警 → 及时响应
5. 备份恢复 → 数据安全
6. 安全加固 → 防攻击

**检查清单**:
- [ ] Docker镜像优化 (<5GB)
- [ ] GPU调度配置正确
- [ ] 监控指标完整
- [ ] 日志集中收集
- [ ] 备份策略测试
- [ ] 压力测试通过
- [ ] 文档完善

生产部署是AIGC系统稳定运行的基础!
