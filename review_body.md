## Code Review — v1.2.0

整体来看 v1.2.0 的 6 个新模块结构清晰、职责分明，代码质量很高。以下是需要修正的问题：

---

### 1. [Critical] core/reports.py:82 — benchmark report 读取已重命名的字段

```python
detected = s.get("anomalies_detected", 0)
```

基准测试（tests/adversarial.py）已将 anomalies_detected 重命名为 anomalies_final_state。但 generate_benchmark_report() 仍读取旧字段名，导致报告中的 detected 计数永远为 0。

同时 recovery_rate_pct 字段也不存在了（已改为 antibody_rate_pct）。

**Fix:** 同步更新字段名。

---

### 2. [Critical] core/adversarial_trainer.py:136-141 — TrainingEvaluator 有同样的测量偏差

```python
has_anomalies = len(result.get("anomalies", [])) > 0
...
detected = has_anomalies or immune_active
```

使用 has_anomalies（最终状态残留）来计算检测分数。同样的根因：validate_antibody_node 清空 anomalies 后，即使系统正确检测并响应了异常，has_anomalies 仍为 False。

**Fix:** 改为以 immune_active 作为检测的主要信号，与基准测试的修复保持一致。

---

### 3. [High] api.py:398-406 — /memory/import 存在路径遍历风险

```python
async def import_memory(path: str):
    ...
    count = memory_db.import_antibodies(path)
```

path 作为 query parameter 直接传入，没有做路径校验。攻击者可以传入 ../../../etc/passwd。

**Fix:** 添加路径白名单校验（限制到 exports/ 目录），或使用 POST body 传递路径。

---

### 4. [High] api.py:364-373 — 同步训练端点阻塞 API

```python
@app.post("/train", ...)
async def start_training(...):
    trainer = AdversarialTrainer(...)
    stats = trainer.train()
```

训练可能持续数分钟，同步执行会阻塞整个 uvicorn worker 线程池。后续请求会排队等待。

**Fix:** 使用 BackgroundTasks + 任务 ID 轮询模式。

---

### 5. [Medium] core/notifications.py:9 — 未使用的 threading 导入

```python
import threading
...
self._lock = threading.Lock()
```

self._lock 被创建但从未获取。send_message 不是线程安全的。

**Fix:** 移除未使用的导入，或在 send_message 中真正使用锁。

---

### 6. [Medium] core/adversarial_trainer.py:178 — 运行时延迟导入有副作用

```python
def train(self, ...):
    from immune_agent import run_single_query
```

immune_agent.py 在模块顶层执行配置校验和 sys.exit(1)（当 API key 缺失时）。如果在无 API key 的环境中调用训练，会直接退出进程。

**Fix:** 将 run_single_query 提取到 core/ 层的独立模块，避免 CLI 层副作用。

---

### 总结

| 等级 | 数量 | 关键修复 |
|------|------|---------|
| Critical | 2 | 字段名同步 + 训练评估器测量偏差 |
| High | 2 | 路径遍历 + API 阻塞 |
| Medium | 2 | 未用锁 + 导入副作用 |

建议合并前至少修复 Critical 级别的两个问题。
