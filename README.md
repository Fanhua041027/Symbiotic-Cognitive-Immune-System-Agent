# Symbiotic Cognitive Immune System Agent

**共生认知免疫系统智能体** — 受生物免疫系统启发的多智能体协作防御框架。
具备**自我诊断、自我修复和自我进化**能力的 AI Agent 框架。

[![CI](https://github.com/Fanhua041027/Symbiotic-Cognitive-Immune-System-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Fanhua041027/Symbiotic-Cognitive-Immune-System-Agent/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 核心思想

| 生物免疫系统 | 本项目对应机制 |
|-------------|---------------|
| 先天免疫 | 规则引擎 + 异常检测基线 |
| 适应性免疫 | 智能体协作学习与模式识别 |
| 免疫记忆 | ChromaDB 持久化向量存储 + 智能 Token 重叠检索 |
| T 细胞 | Monitor Agent — 监察主智能体的执行过程 |
| B 细胞 / 抗体 | Antibody Generator — 发现异常后自动生成修复补丁 |
| 抗原识别 | Worker 自检 + Monitor 双重异常检测 |

## 智能体工作流程

```
用户输入 → [Worker 主智能体] → [Monitor 监察员 T 细胞]
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
                       重新执行（携带抗体 + 历史记忆）

失败 ≥N 次 → [人类告警升级] → 生成 JSON 报告
```

## 特性

| 特性 | 说明 |
|------|------|
| **自诊断** | Worker 三段式自检 + Monitor T 细胞 5 维度深度分析（含 severity 评级） |
| **自修复** | LLM 驱动抗体生成，自动产生带终止守卫的修复补丁 |
| **多级沙箱** | simulated（启发式关键词）/ ast（Python AST 静态分析）/ docker（容器执行） |
| **持久记忆** | ChromaDB 持久化存储，自动 InMemoryStore 回退，跨会话复用抗体 |
| **抗体去重** | Jaccard Token 相似度 >= 0.7 自动跳过，防内存膨胀 |
| **执行追踪** | 完整 workflow_trace 记录，Web UI 可视化彩色执行路径 |
| **多 Provider** | 支持 OpenAI / DeepSeek / 自定义 OpenAI-compatible 端点 |
| **Web UI** | 6 标签页 Streamlit 界面：查询 / 历史 / 记忆 / 流程图 / 基准测试 / 指标 |
| **REST API** | FastAPI 端点：health / query / stats / memory CRUD / config / demo |
| **系统指标** | 实时追踪成功率、P95 延迟、异常分析、免疫激活率 |
| **配置持久化** | Web UI 直接编辑并保存 .env 配置，白名单安全校验 |
| **告警升级** | 连续失败 >= N 次自动生成人类可读 JSON 报告 |
| **Git 自动备份** | 免疫响应触发时自动创建 Git checkpoint |
| **对抗测试** | 内置 12 个对抗测试用例，量化免疫系统效能 |
| **容器化** | Docker / docker-compose 一键部署 |
| **零依赖运行** | 无 chromadb 时自动降级为内存 Token 重叠搜索 |

## 快速开始

### 1. 安装

```bash
# 基础依赖
pip install -r requirements.txt

# Web UI（推荐）
pip install streamlit

# 持久化免疫记忆（可选，默认使用内存回退）
pip install chromadb

# REST API（可选）
pip install fastapi uvicorn

# 开发
pip install pytest
```

或一键安装全部：

```bash
pip install -r requirements.txt streamlit chromadb fastapi uvicorn pytest
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，支持 OpenAI / DeepSeek / Custom
```

### 3. 运行

```bash
# 内置演示（自动触发认知异常并观察免疫响应）
python immune_agent.py

# 自定义查询
python immune_agent.py -q "Write a function with an infinite loop"

# 交互模式
python immune_agent.py -i

# 系统统计
python immune_agent.py -s

# Web UI（推荐）
streamlit run app.py

# REST API
python api.py
```

### 4. 运行测试

```bash
# 148+ 个单元测试（无需 API Key）
python -m pytest tests/ -v --tb=short

# 12 个对抗性测试（需要 API Key）
python immune_agent.py --benchmark
```

### 5. Docker 部署

```bash
docker compose build
docker compose run --rm immune-agent
```

## 项目结构

```
├── immune_agent.py          # CLI 入口（demo / interactive / single-query）
├── app.py                   # Streamlit Web UI（6 标签页）
├── api.py                   # FastAPI REST API
├── setup_project.py         # 项目初始化向导
├── Makefile                 # 常用命令快捷方式
├── Dockerfile               # Docker 构建
├── docker-compose.yml       # Docker 编排
├── pyproject.toml           # 项目元数据 + 可选依赖组
├── requirements.txt         # 核心依赖
├── .env.example             # 环境变量模板
├── LICENSE                  # MIT 许可证
├── CONTRIBUTING.md          # 贡献指南
├── CHANGELOG.md             # 版本历史
├── SECURITY.md              # 安全策略
│
├── .github/
│   ├── workflows/ci.yml     # CI 流水线
│   ├── dependabot.yml       # 自动依赖更新
│   └── ISSUE_TEMPLATE/      # Issue 模板（bug / feature）
│
├── core/
│   ├── config.py            # 配置验证与集中管理 + .env 持久化
│   ├── logger.py            # 结构化日志（控制台 + 滚动文件）
│   ├── state.py             # ImmunologyState TypedDict
│   ├── memory.py            # 免疫记忆（ChromaDB / 内存回退 + 去重）
│   ├── nodes.py             # 四个核心智能体节点 + git 自动备份
│   ├── workflow.py          # LangGraph 工作流 + 执行轨迹
│   ├── sandbox.py           # 多级沙箱验证（simulated / ast / docker / e2b）
│   ├── escalation.py        # 人类告警升级系统
│   ├── metrics.py           # 指标追踪（成功率 / 延迟 / 异常分析）
│   └── viz.py               # 工作流可视化
│
├── tests/
│   ├── test_core.py         # 148+ 个单元测试
│   ├── test_api.py          # FastAPI 端点集成测试
│   └── adversarial.py       # 12 个对抗性测试用例
│
├── logs/                    # 运行日志（自动创建）
├── .immune_db/              # ChromaDB 持久化数据（自动创建）
├── escalations/             # 告警升级报告（自动创建）
├── benchmarks/              # 对抗测试报告（自动创建）
└── metrics/                 # 指标报告（自动创建）
```

## Web UI

六标签页界面，通过 `streamlit run app.py` 启动：

| 标签页 | 功能 |
|--------|------|
| **Run Query** | 系统状态面板 + 查询输入 + 执行轨迹可视化 + 结果详情 |
| **History** | 会话查询历史，支持清空和 JSON 导出 |
| **Memory** | 抗体浏览器，支持搜索、删除、清空 |
| **Workflow Graph** | Mermaid 流程图 + ASCII 架构图 |
| **Benchmark** | 一键运行 12 个对抗测试，实时进度和统计 |
| **Metrics** | 成功率 / 异常率 / P95 延迟 / 异常来源分析 / 报告导出 |

## REST API

通过 `python api.py` 启动（默认端口 8000）：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 + 配置摘要 |
| `/query` | POST | 执行单次查询 |
| `/stats` | GET | 系统指标统计 |
| `/memory` | GET | 列出抗体记忆 |
| `/memory/{id}` | DELETE | 删除指定抗体 |
| `/memory` | DELETE | 清空所有抗体 |
| `/config` | PATCH | 更新配置 |
| `/demo` | GET | 列出内置演示 |
| `/demo/{name}` | POST | 运行指定演示 |

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
| `python api.py` | 启动 REST API |
| `python setup_project.py` | 初始化向导 |

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

## 多 Provider 配置

```bash
# DeepSeek
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-xxx
MAIN_LLM_MODEL=deepseek-chat

# 自定义 OpenAI-compatible 端点
LLM_PROVIDER=custom
CUSTOM_API_KEY=sk-xxx
CUSTOM_API_BASE=https://your-endpoint/v1
```

## 对抗测试

内置 12 个对抗测试用例，覆盖：

- 无限循环陷阱
- 逻辑矛盾
- 自指悖论
- 模糊需求
- 幻觉诱导
- 矛盾约束
- 递归缺失基准条件

每条用例独立运行，系统自动统计异常检测率、抗体生成率、免疫恢复率。

## Docker 沙箱模式

```bash
echo "SANDBOX_MODE=docker" >> .env
python immune_agent.py
```

## 技术栈

- **LangGraph** — 状态图工作流编排
- **LangChain / ChatOpenAI** — LLM 统一调用接口
- **ChromaDB** — 向量持久化存储
- **Streamlit** — Web UI
- **FastAPI + uvicorn** — REST API
- **Python AST** — 代码静态安全分析

## 许可证

MIT
