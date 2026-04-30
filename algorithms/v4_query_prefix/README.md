# v4: query 前缀对齐（探索性）

## 你的判断

> 关于 query 的前缀问题可以尝试，但我认为这不是关键问题

同意。这条不是必要修复，只做探索。

## 背景

`hierarchical_manager.py:403-405`：
```python
def _make_level_aware_text(self, level, content):
    return f"{level.name}: {content}"
```
节点的 `level_embedding` 是文本 `"DOMAIN: Education..."` 的 embedding。

`base_retriever.py:_prepare_query_embedding` 直接：
```python
return list(self.embedding_function(query_text))
```
query **没有加任何前缀**。如果训练时节点用的是 `level_embedding`（即 `use_level_embedding=True`），推理时 query 的 embedding 和节点的 embedding 落在不同的分布上。

## 修改

给 `BaseHierarchicalRetriever` 加一个 `query_prefix` 参数：

```python
def __init__(self, ..., query_prefix: Optional[str] = None):
    ...
    self.query_prefix = query_prefix or ""

def _prepare_query_embedding(self, query_text, query_embedding):
    if query_embedding is not None:
        return list(query_embedding)
    if query_text is None:
        raise ValueError(...)
    prefixed = f"{self.query_prefix}{query_text}" if self.query_prefix else query_text
    return list(self.embedding_function(prefixed))
```

使用：**通过属性设置**（推荐），因为 `BaseHyperbolicRetriever.__init__` 不在 v4 的修改范围内，构造时直接传 `query_prefix` 会因 `**kwargs` 转发到不接受该参数的签名而报错。

```python
retriever = MultiParentAngularHyperbolicRetriever(
    vector_store=vs,
    checkpoint_path=ckpt,
)
retriever.query_prefix = "QUERY: "    # 直接设属性
```

`CosineRetriever` 因为直接继承 `HierarchicalRetrieverBase`，构造时可以直接传：
```python
# CosineRetriever 支持构造参数
# 但需要 CosineRetriever.__init__ 也 accept query_prefix — v4 默认不改它
# 所以统一用属性设置更保险
```

可以扫 `["", "QUERY: ", "DOMAIN: ", "CATEGORY: ", "KEYWORD: ", "DIALOGUE: "]` 这些候选，看哪种最匹配节点分布。`diagnose.py` 里已经按属性方式设置（`inf.retriever.query_prefix = ...`）。

## 更进一步：分层前缀（可选，注释掉的代码里有）

给 `BaseHyperbolicRetriever.retrieve` 加一个 `query_prefix_per_level` 选项：每一层检索前按当前层级重新算 query embedding，喂 `"DOMAIN: {q}"` / `"CATEGORY: {q}"` 等。这需要在每层 walk 里重新算 query，成本略高但最对齐。

本版本**只实现静态前缀**。分层前缀作为注释留在代码里作为备选。

## 涉及文件

- `model/retrievers/base_retriever.py`

## 如何应用

```bash
python algorithms/apply.py v4_query_prefix
```

然后在代码里构造 retriever 时传 `query_prefix`，或修改 `llm_inference.py` 相应位置。

## 如何验证

**诊断实验**：先看 query embedding 和节点 level_embedding 的分布距离。

```python
# 单独跑一下
from model.encoders import EmbeddingEncoder
enc = EmbeddingEncoder()

import numpy as np
q_raw = np.array(enc.generate_embedding("When did Caroline go to the LGBTQ support group?"))
q_pref = np.array(enc.generate_embedding("QUERY: When did Caroline go to the LGBTQ support group?"))
q_dom  = np.array(enc.generate_embedding("DOMAIN: When did Caroline go to the LGBTQ support group?"))

# 和某个节点的 level_embedding 比较
node_raw = np.array(enc.generate_embedding("Social Issues"))
node_lvl = np.array(enc.generate_embedding("DOMAIN: Social Issues"))

def cos(a, b): return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("q_raw vs node_raw:", cos(q_raw, node_raw))
print("q_raw vs node_lvl:", cos(q_raw, node_lvl))
print("q_pref vs node_lvl:", cos(q_pref, node_lvl))
print("q_dom vs node_lvl:", cos(q_dom, node_lvl))
```

若 `q_dom vs node_lvl > q_raw vs node_lvl`，前缀能把 query 拉近节点分布，值得尝试。

## 需不需要重训？

**不需要**。纯推理侧改动。

## 局限性

- sentence encoder 对前缀的处理取决于模型本身。MiniLM 类小模型可能对短前缀（几个 token）吸收不明显。
- 分层前缀方案（注释里）成本略高，用户按需启用。

## 不影响其他版本

只改 `base_retriever.py` 的 `__init__` 和 `_prepare_query_embedding`。完全向后兼容（不传 `query_prefix` 就是原行为）。
