# Symbiotic Cognitive Immune System Agent

**共生认知免疫系统智能体** — 受生物免疫系统启发的多智能体协作防御框架。
具备**自我诊断、自我修复和自我进化**能力的 AI Agent 框架。

## 核心思想

| 生物免疫系统 | 本项目对应机制 |
|-------------|---------------|
| 先天免疫 | 规则引擎 + 异常检测基线 |
| 适应性免疫 | 智能体协作学习与模式识别 |
| 免疫记忆 | ChromaDB 向量存储 + 历史抗体缓存 |
| T 细胞 | Monitor Agent — 监察主智能体的执行过程 |
| B 细胞 / 抗体 | Antibody Generator — 发现异常后自动生成代码补丁 |
| 抗原识别 | 输入异常检测与分类 |

## 智能体工作流程

```
用户输入 → [Worker 主智能体] → [Monitor 监察员]
                                      │
                       ┌────────────────┼────────────────┐
                       ▼                ▼                ▼
                   健康 (END)      发现异常           继续执行
                                      │
                            [Antibody 生成器]
                                      │
                            [Sandbox 验证器] ← (simulated / ast / docker)
                                      │
                              ┌───────┘
                              ▼
                       重新执行（携带抗体）

失败 ≥N 次 → [人类告警升级] → 生成 JSON 报告
```

## 核心特性

| 特性 | 说明 |
|------|------|
| **自诊断** | Worker 自检 + Monitor T 细胞双重异常检测 |
| **自修复** | LLM 驱动抗体生成，自动产生修复补丁 |
| **多级沙箱** | simulated（启发式）/ ast（静态分析）/ docker（容器执行） |
| **持久记忆** | ChromaDB 持久化存储，跨会话复用抗体 |
| **多 Provider** | 支持 OpenAI / DeepSeek / 自定义 OpenAI-compatible 端点 |
| **Web UI** | Streamlit 可视化界面（查询/图表/基准测试） |
| **告警升级** | 连续失败 ≥N 次自动生成人类可读的 JSON 报告 |
| **对抗测试** | 内置 12 个对抗测试用例，量化免疫系统效能 |
| **容器化** | Docker / docker-compose 一键部署 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install streamlit  # Web UI（可选）
pip install chromadb   # 持久化记忆（可选，默认使用内存回退）
pip install pytest     # 运行测试（可选）
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key
# 支持 OpenAI / DeepSeek / Custom 三种 Provider
```

### 3. 运行

```bash
# 运行内置演示（自动触发认知异常并观察免疫响应）
python immune_agent.py

# 自定义查询
python immune_agent.py --query "Write a function with an infinite loop"

# 交互模式
python immune_agent.py --interactive

# 查看系统统计
python immune_agent.py --stats

# Web UI（需要 streamlit）
streamlit run app.py
```

### 4. 运行测试

```bash
# 34 个单元测试（无需 API Key）
python -m pytest tests/ -v --tb=short

# 12 个对抗性测试基准（需要 API Key）
python tests/adversarial.py
python immune_agent.py --benchmark
```

### 5. Docker 部署

```bash
docker compose build
docker compose run --rm immune-agent
```

## 项目结构

```
├── app.py                          # Streamlit Web UI
├── immune_agent.py                 # 主入口 / CLI
├── Makefile                        # 常用命令快捷方式
├── Dockerfile                      # Docker 构建文件
├── docker-compose.yml              # Docker 编排
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── core/
│   ├── __init__.py
│   ├── config.py                   # 配置验证与集中管理
│   ├── logger.py                   # 结构化日志（控制台 + 文件滚动）
│   ├── state.py                    # ImmunologyState 状态定义
│   ├── memory.py                   # 免疫记忆（ChromaDB / 内存回退）
│   ├── nodes.py                    # 四个核心智能体节点
│   ├── workflow.py                 # LangGraph 工作流图
│   ├── sandbox.py                  # 多级沙箱验证（simulated/ast/docker）
│   ├── escalation.py               # 人类告警升级系统
│   └── viz.py                      # 工作流可视化
│
├── tests/
│   ├── __init__.py
│   ├── test_core.py                # 34 个单元测试
│   └── adversarial.py              # 12 个对抗性测试用例
│
├── logs/                           # 运行日志（自动创建）
├── .immune_db/                     # ChromaDB 持久化数据（自动创建）
├── escalations/                    # 告警升级报告（自动创建）
└── benchmarks/                     # 对抗测试报告（自动创建）
```

## 配置项

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_PROVIDER` | `openai` | Provider: openai / deepseek / custom |
| `OPENAI_API_KEY` | — | OpenAI API 密钥 |
| `DEEPSEEK_API_KEY` | — | DeepSeek API 密钥 |
| `CUSTOM_API_KEY` | — | 自定义端点 API 密钥 |
| `CUSTOM_API_BASE` | — | 自定义端点 Base URL |
| `MAIN_LLM_MODEL` | `gpt-4o` | Worker 模型 |
| `MONITOR_LLM_MODEL` | `gpt-4o-mini` | Monitor 模型 |
| `ANTIBODY_LLM_MODEL` | `gpt-4o` | 抗体生成模型 |
| `LLM_TEMPERATURE` | `0.7` | LLM 温度参数 |
| `MAX_ITERATIONS` | `5` | 最大免疫迭代次数 |
| `SANDBOX_MODE` | `simulated` | 沙箱模式: simulated / ast / docker |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `ESCALATION_THRESHOLD` | `3` | 连续失败告警阈值 |

## CLI 命令

| 命令 | 说明 |
|------|------|
| `python immune_agent.py` | 运行内置演示 |
| `python immune_agent.py -q "..."` | 单次查询 |
| `python immune_agent.py -i` | 交互模式 |
| `python immune_agent.py -b` | 对抗测试基准 |
| `python immune_agent.py -g` | 显示工作流图 |
| `python immune_agent.py -s` | 查看系统统计 |
| `python immune_agent.py -q "..." -j` | JSON 格式输出 |
| `python immune_agent.py -q "..." -t 30` | 30 秒超时查询 |
| `streamlit run app.py` | 启动 Web UI |

## 对抗测试

内置 12 个对抗测试用例，覆盖：
- 无限循环陷阱
- 逻辑矛盾
- 自指悖论
- 模糊需求
- 幻觉诱导
- 矛盾约束

每条用例独立运行，系统会统计异常检测率、抗体生成率、免疫恢复率和人类告警次数。

## Docker 沙箱模式

```bash
echo "SANDBOX_MODE=docker" >> .env
python immune_agent.py
```

## 多 Provider 用法

```bash
# DeepSeek
echo "LLM_PROVIDER=deepseek" >> .env
echo "DEEPSEEK_API_KEY=sk-xxx" >> .env
echo "MAIN_LLM_MODEL=deepseek-chat" >> .env

# 自定义 OpenAI-compatible 端点
echo "LLM_PROVIDER=custom" >> .env
echo "CUSTOM_API_KEY=sk-xxx" >> .env
echo "CUSTOM_API_BASE=https://your-endpoint/v1" >> .env
```

## 许可证

MIT
