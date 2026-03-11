# -*- coding: utf-8 -*-
"""
每日算法+AI手撕练习脚本 v2.0
功能：
1. 新题+旧题混合练习（30%旧题，70%新题）
2. 进度永远累加
3. 练习记录保存到文档
"""

import os
import re
import random
import json
import sys
from datetime import datetime

# 设置UTF-8输出
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ============ 配置 ============
DATA_DIR = "D:/2026_project"
LEETCODE_EASY_FILE = os.path.join(DATA_DIR, "Hot100_Easy.md")
LEETCODE_MEDIUM_FILE = os.path.join(DATA_DIR, "Hot100_Medium.md")
PROGRESS_FILE = os.path.join(DATA_DIR, "daily_progress.json")
RECORD_FILE = os.path.join(DATA_DIR, "practice_record.md")
OLD_RATIO = 0.3  # 旧题比例 30%

# ============ LLM面试知识点库 ============
LLM_KNOWLEDGE = {
    "基础概念": [
        {
            "title": "Transformer基本结构",
            "content": """Transformer是一种基于自注意力机制的深度学习架构，由encoder和decoder两部分组成。

核心组件：
1. 多头注意力(Multi-Head Attention)：并行计算多个注意力头，捕获不同位置的不同表示子空间信息
2. 前馈网络(Feed Forward)：两层全连接网络，对每个位置分别进行非线性变换
3. 残差连接(Residual Connection)：缓解梯度消失问题
4. 层归一化(Layer Normalization)：稳定训练

Q: Transformer中使用的是什么注意力机制？
A: 自注意力机制(Self-Attention)，允许序列中的每个位置关注序列中的所有其他位置。

Q: 为什么需要多头注意力？
A: 多个头可以关注不同位置的不同表示子空间，捕获不同类型的依赖关系。""",
            "blanks": [
                ("自注意力机制", "允许序列中的每个位置关注序列中的所有其他位置"),
                ("多头", "并行计算多个注意力头，捕获不同位置的不同表示子空间信息")
            ]
        },
        {
            "title": "GPT vs BERT 区别",
            "content": """GPT和BERT是两种不同的大语言模型架构：

GPT (Generative Pre-trained Transformer)：
- 单向语言模型：只使用左侧上文来预测下一个词
- 适合：文本生成任务
- 特点：自回归模型

BERT (Bidirectional Encoder Representations from Transformers)：
- 双向模型：同时使用左右上下文
- 适合：理解任务（分类、问答、NER）
- 特点：使用MLM(掩码语言模型)预训练

Q: BERT为什么使用双向注意力？
A: BERT的预训练任务是掩码语言模型，需要同时看到左右上下文来预测被mask的词。

Q: GPT是自回归还是自编码？
A: 自回归模型。""",
            "blanks": [
                ("单向", "只使用左侧上文来预测下一个词"),
                ("双向", "同时使用左右上下文")
            ]
        },
        {
            "title": "位置编码(Positional Encoding)",
            "content": """Transformer没有循环结构，需要显式添加位置信息。

主要方法：
1. 绝对位置编码：为每个位置学习一个向量
2. 相对位置编码：考虑token之间的相对距离

Sinusoidal位置编码：
- 使用正弦和余弦函数生成位置向量
- 公式：PE(pos,2i) = sin(pos/10000^(2i/d_model))
- 公式：PE(pos,2i+1) = cos(pos/10000^(2i/d_model))

Q: 为什么使用sin和cos函数？
A: 可以让模型学习相对位置关系，因为sin(a+b)和cos(a+b)可以用sin a/cos a的线性组合表示。""",
            "blanks": [
                ("正弦和余弦函数", "生成位置向量"),
                ("相对位置关系", "让模型学习token之间的相对距离")
            ]
        }
    ],
    "训练技术": [
        {
            "title": "LoRA微调原理",
            "content": """LoRA (Low-Rank Adaptation) 是一种高效的微调方法：

核心思想：
- 冻结预训练模型的权重
- 在Transformer的注意力层添加低秩分解矩阵
- 只训练新增的低秩矩阵

具体实现：
- 对于原始权重矩阵 W (d×k)
- 添加更新：W' = W + BA
- 其中 B (d×r), A (r×k)，r << min(d,k)
- 训练参数量从 d×k 减少到 d×r + r×k

Q: LoRA的秩(rank)一般设置多少？
A: 通常4-32，r越大效果越好但参数量越多。""",
            "blanks": [
                ("低秩分解", "在注意力层添加低秩矩阵"),
                ("W + BA", "原始权重加上低秩更新")
            ]
        },
        {
            "title": "RLHF训练流程",
            "content": """RLHF (Reinforcement Learning from Human Feedback) 三阶段：

1. 预训练语言模型
   - 使用大规模文本进行无监督预训练
   - 得到基座模型

2. 训练奖励模型(Reward Model)
   - 收集人类偏好数据
   - 训练模型学习人类偏好

3. 强化学习微调
   - 使用PPO算法优化策略
   - 奖励模型提供reward signal
   - 加入KL散度约束防止偏离基座

Q: RLHF为什么用PPO？
A: PPO是一种稳定且sample-efficient的policy gradient算法，适合大模型训练。""",
            "blanks": [
                ("PPO", "强化学习微调阶段使用的算法"),
                ("奖励模型", "学习人类偏好，提供reward signal")
            ]
        },
        {
            "title": "梯度消失与爆炸",
            "content": """深度学习中的经典问题：

梯度消失：
- 原因：链式法则导致梯度指数级衰减
- 表现：深层网络前面层参数几乎不更新
- 解决：残差连接、层归一化、ReLU激活

梯度爆炸：
- 原因：梯度指数级增长
- 表现：参数更新过大，训练不稳定
- 解决：梯度裁剪(gradient clipping)

Q: Transformer中如何缓解梯度问题？
A: 使用残差连接(Residual Connection)和层归一化(Layer Normalization)。""",
            "blanks": [
                ("残差连接和层归一化", "缓解Transformer中的梯度问题"),
                ("梯度裁剪", "防止梯度爆炸")
            ]
        }
    ],
    "推理优化": [
        {
            "title": "KV Cache加速推理",
            "content": """KV Cache是大模型推理优化的核心技术：

问题：自回归生成中，每个token生成都需要重新计算之前所有token的K、V矩阵，效率低。

解决方案：
- 缓存已计算的K(key)和V(value)矩阵
- 只计算新token的Q、K、V
- 每次只做一次注意力计算

Q: KV Cache会占用多少显存？
A: 约等于 2 * batch_size * num_layers * seq_len * hidden_dim * 2 bytes

例如：LLaMA-7B, batch=1, seq_len=2048, layers=32, hidden=4096
约 2 * 1 * 32 * 2048 * 4096 * 2 ≈ 1GB""",
            "blanks": [
                ("缓存已计算的K和V", "避免重复计算"),
                ("只计算新token", "提高推理效率")
            ]
        },
        {
            "title": "Beam Search解码",
            "content": """Beam Search是常用的解码策略：

Greedy解码：每次选择概率最高的token
- 缺点：可能陷入局部最优

Beam Search：
- 保留top-k个最可能的下一步
- 继续扩展直到结束
- 选择整体概率最高的序列
- 参数：beam_size

Q: beam_size=1时是什么？
A: 就是greedy解码。

Q: 为什么beam search比greedy好？
A: 考虑了全局最优，而不是只看当前一步。""",
            "blanks": [
                ("top-k个最可能的下一步", "Beam Search的核心"),
                ("全局最优", "考虑整体序列概率而不是单步")
            ]
        }
    ],
    "RAG相关": [
        {
            "title": "RAG工作流程",
            "content": """RAG (Retrieval-Augmented Generation) = 检索 + 生成

工作流程：
1. 索引阶段：
   - 文档切分
   - 向量化(embedding)
   - 存入向量数据库

2. 查询阶段：
   - 用户问题向量化
   - 相似度检索(top-k)
   - 上下文拼接

3. 生成阶段：
   - LLM根据检索到的上下文生成答案

Q: 为什么要用RAG？
A: 1. 解决知识过时问题
   2. 减少幻觉
   3. 降低成本(不用全量微调)""",
            "blanks": [
                ("检索+生成", "RAG的核心"),
                ("向量化", "将文本转换为向量用于相似度计算")
            ]
        },
        {
            "title": "向量检索方法",
            "content": """常见向量检索方法：

1. 暴力搜索(Exact Search)
   - 计算query与所有向量的距离
   - 优点：准确
   - 缺点：O(n)复杂度

2. 近似最近邻(ANN)
   - HNSW：基于图的索引
   - IVF + PQ：倒排索引+乘积量化
   - 优点：速度快
   - 缺点：有损

Q: 生产环境常用什么？
A: 常用Faiss、Milvus、Qdrant等向量数据库，内部使用ANN算法。""",
            "blanks": [
                ("HNSW", "基于图的近似最近邻算法"),
                ("Faiss", "Facebook开源的向量检索库")
            ]
        }
    ],

    # 新增：深入技术
    "模型细节": [
        {
            "title": "Self-Attention计算公式",
            "content": """Self-Attention是Transformer的核心，其计算过程：

公式：Attention(Q, K, V) = softmax(QK^T / √d_k)V

其中：
- Q (Query): 当前词的查询向量
- K (Key): 所有词的键向量
- V (Value): 所有词的值向量
- d_k: 向量维度

计算步骤：
1. Q、K、V生成：输入X分别乘以三个权重矩阵W_Q、W_K、W_V
2. 相似度计算：score = QK^T
3. 缩放：score = score / √d_k（防止梯度爆炸）
4. Softmax：attention_weights = softmax(score)
5. 加权求和：output = attention_weights * V

Q: 为什么要除以√d_k？
A: 当d_k较大时，点积结果会很大，导致softmax进入梯度很小的区域。除以√d_k可以缩放到合理的范围。

Q: 多头注意力是什么？
A: 多个注意力头并行计算，每个头学习不同的表示子空间，最后concat起来。""",
            "blanks": [
                ("QK^T / √d_k", "缩放点积注意力"),
                ("softmax", "将分数转换为概率分布")
            ]
        },
        {
            "title": "Layer Normalization vs Batch Normalization",
            "content": """Layer Norm和Batch Norm的区别：

Batch Normalization：
- 对batch维度做归一化
- 依赖于batch size，batch小时不稳定
- 适合CV，不适合NLP

Layer Normalization：
- 对单个样本做归一化
- 不依赖于batch size
- 适合NLP，适合Transformer

在Transformer中：
- 每个样本独立处理（因为是位置编码决定顺序）
- Layer Norm在每个Transformer块中
- 典型顺序：残差连接 → Layer Norm

公式：LN(x) = γ * (x - μ) / σ + β
其中 μ = mean(x), σ = std(x)

Q: Transformer为什么用Layer Norm？
A: NLP中序列长度可变，不同位置的token重要性不同，Layer Norm更稳定。""",
            "blanks": [
                ("对单个样本做归一化", "Layer Normalization"),
                ("对batch维度做归一化", "Batch Normalization")
            ]
        },
        {
            "title": "梯度消失与爆炸详解",
            "content": """深度学习中的梯度问题：

梯度消失原因：
- 链式法则连乘导致梯度指数级衰减
- sigmoid导数最大0.25
- 网络越深，前面层更新越慢

梯度爆炸原因：
- 梯度连乘导致指数级增长
- 参数初始化过大
- 权重矩阵的奇异值>1

解决方案：
1. 残差连接：y = F(x) + x，梯度可直接回传
2. 层归一化：稳定每层输入分布
3. 激活函数：ReLU导数为1，缓解消失
4. 梯度裁剪：threshold截断
5. 初始化：Xavier/Kaiming初始化

Q: Transformer中如何解决？
A: 主要靠残差连接(Residual Connection)和层归一化(Layer Normalization)。

Q: 为什么ReLU更好？
A: 正区间导数为1，不会缩小梯度。""",
            "blanks": [
                ("残差连接和层归一化", "Transformer解决梯度问题的方法"),
                ("梯度裁剪", "防止梯度爆炸")
            ]
        }
    ],

    "训练进阶": [
        {
            "title": "Adam优化器原理",
            "content": """Adam (Adaptive Moment Estimation) 是最常用的优化器：

核心思想：
- 一阶矩估计：梯度均值（类似动量）
- 二阶矩估计：梯度方差（自适应学习率）

算法步骤：
1. 初始化：m=0, v=0, t=0
2. 每步：
   - t = t + 1
   - g = ∇θ L (梯度)
   - m = β₁*m + (1-β₁)*g (一阶矩)
   - v = β₂*v + (1-β₂)*g² (二阶矩)
   - m̂ = m / (1-β₁ᵗ) (偏差校正)
   - v̂ = v / (1-β₂ᵗ) (偏差校正)
   - θ = θ - α * m̂ / (√v̂ + ε)

超参数：
- α: 学习率 (通常0.001)
- β₁: 0.9 (一阶矩衰减)
- β₂: 0.999 (二阶矩衰减)
- ε: 10⁻⁸ (数值稳定)

Q: 为什么要偏差校正？
A: 初期m和v接近0，需要校正。""",
            "blanks": [
                ("一阶矩估计", "梯度的指数移动平均"),
                ("二阶矩估计", "梯度平方的指数移动平均")
            ]
        },
        {
            "title": "学习率调度策略",
            "content": """学习率调度（Learning Rate Scheduling）：

常见策略：
1. Warmup：开始时学习率从小到大
   - 原因：初期参数随机，大的lr会不稳定
   - 常用：linear warmup

2. Cosine Annealing：余弦退火
   - lr = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(π * t / T))
   - 平滑下降，后期接近0

3. Step Decay：阶梯下降
   - 每N个epoch降低学习率

4. Exponential Decay：指数下降
   - lr = lr * γ^t

5. Polynomial Decay：多项式下降
   - lr = (lr_max - lr_min)*(1 - t/T)^p + lr_min

Transformer常用：
- Warmup + Cosine Annealing
- 例如：前10%步warmup，剩余余弦退火到0

Q: 为什么要warmup？
A: 初期参数随机，小lr更稳定。""",
            "blanks": [
                ("warmup", "初期学习率从小到大"),
                ("cosine annealing", "余弦曲线下降")
            ]
        },
        {
            "title": "混合精度训练",
            "content": """混合精度训练（Mixed Precision Training）：

背景：
- FP32(32位浮点)精度高但慢
- FP16(16位浮点)快但可能溢出

解决方案：混合精度

关键组件：
1. FP16计算：前向和反向用FP16
2. FP32权重：主权重存FP32
3. Loss Scaling：放大loss防止下溢

实现流程：
1. 权重拷贝：FP32 → FP16
2. 前向：FP16计算
3. Loss：FP16计算loss
4. Loss放大：loss * scale
5. 反向：FP16梯度
6. 梯度缩小：grad / scale
7. 更新：FP32权重

优点：
- 2倍加速
- 减少显存

Q: 为什么要FP32存权重？
A: 梯度累积时FP16可能下溢。""",
            "blanks": [
                ("FP16计算，FP32权重", "混合精度的核心"),
                ("Loss Scaling", "防止FP16下溢")
            ]
        }
    ]
}


class DailyPracticeApp:
    def __init__(self):
        self.questions = []
        self.current_index = 0
        self.llm_topics = self._flatten_llm_topics()
        self.progress = self._load_progress()

    def _flatten_llm_topics(self):
        topics = []
        for category, items in LLM_KNOWLEDGE.items():
            for item in items:
                topics.append({
                    'category': category,
                    'title': item['title'],
                    'content': item['content'],
                    'blanks': item['blanks'],
                    'id': f"llm_{item['title']}"
                })
        return topics

    def _load_progress(self):
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'leetcode_history': [],  # 所有做过的题目ID
            'llm_history': [],        # 所有做过的LLM知识点
            'leetcode_counts': {},   # 每道题练习次数
            'llm_counts': {},        # 每个知识点练习次数
            'total_days': 0          # 总练习天数
        }

    def _save_progress(self):
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, ensure_ascii=False, indent=2)

    def parse_leetcode_file(self, filepath):
        if not os.path.exists(filepath):
            return []

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        questions = []
        parts = re.split(r'(?=### \[)', content)

        for part in parts:
            if not part.strip() or not part.startswith('###'):
                continue

            title_match = re.search(r'###\s*\[(\d+)\.\s*(.+?)\]', part)
            if not title_match:
                continue

            number = title_match.group(1)
            name = title_match.group(2)

            desc_match = re.search(r'#### 题目描述\s*(.+?)(?=####|$)', part, re.DOTALL)
            description = desc_match.group(1).strip() if desc_match else ""

            thought_match = re.search(r'#### 核心思路\s*(.+?)(?=####|$)', part, re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else ""

            code_match = re.search(r'```python\s*(.+?)```', part, re.DOTALL)
            code = code_match.group(1).strip() if code_match else ""

            link_match = re.search(r'\[.+?\]\((.+?)\)', part)
            link = link_match.group(1) if link_match else ""

            questions.append({
                'number': number,
                'name': name,
                'description': description,
                'thought': thought,
                'code': code,
                'link': link,
                'id': f"lc_{number}"
            })

        return questions

    def load_all_questions(self):
        """加载所有题目"""
        self.questions = []
        self.questions.extend(self.parse_leetcode_file(LEETCODE_EASY_FILE))
        self.questions.extend(self.parse_leetcode_file(LEETCODE_MEDIUM_FILE))
        self.questions.sort(key=lambda x: int(x['number']))
        return self.questions

    def select_questions(self, count=2):
        """选择题目：新题+旧题混合"""
        all_questions = self.load_all_questions()
        if not all_questions:
            return []

        history = self.progress.get('leetcode_history', [])

        if not history:
            # 第一次练习，随机选新题
            return random.sample(all_questions, min(count, len(all_questions)))

        # 分离新题和旧题
        new_questions = [q for q in all_questions if q['id'] not in history]
        old_questions = [q for q in all_questions if q['id'] in history]

        # 计算数量
        num_old = int(count * OLD_RATIO)
        num_new = count - num_old

        selected = []

        # 选旧题（优先选练习次数少的）
        if old_questions and num_old > 0:
            # 按练习次数排序，优先选少的
            old_sorted = sorted(old_questions,
                              key=lambda q: self.progress['leetcode_counts'].get(q['id'], 0))
            selected.extend(random.sample(old_sorted, min(num_old, len(old_sorted))))

        # 选新题
        if new_questions and num_new > 0:
            selected.extend(random.sample(new_questions, min(num_new, len(new_questions))))

        # 如果新题不够，用旧题补
        if len(selected) < count:
            remaining = [q for q in all_questions if q not in selected]
            selected.extend(random.sample(remaining, count - len(selected)))

        random.shuffle(selected)
        return selected[:count]

    def select_llm_topic(self):
        """选择LLM知识点：新题+旧题混合"""
        history = self.progress.get('llm_history', [])

        if not history:
            return random.choice(self.llm_topics)

        new_topics = [t for t in self.llm_topics if t['id'] not in history]
        old_topics = [t for t in self.llm_topics if t['id'] in history]

        # 70%新题，30%旧题
        if new_topics and random.random() > OLD_RATIO:
            return random.choice(new_topics)
        elif old_topics:
            # 旧题优先选练习次数少的
            old_sorted = sorted(old_topics,
                              key=lambda t: self.progress['llm_counts'].get(t['id'], 0))
            return random.choice(old_sorted[:3]) if len(old_sorted) > 3 else random.choice(old_sorted)
        else:
            return random.choice(self.llm_topics)

    def create_fill_blanks(self, code, num_blanks=2):
        """创建挖空版本 - 只挖核心逻辑行"""
        # 完全跳过的行首模式
        skip_start = ['class ', 'def ', 'import ', 'return', 'if __name__', '#', '"""', "'''", 'self.']

        lines = code.split('\n')
        # 找出可以挖空的行
        blankable = []

        for idx, line in enumerate(lines):
            stripped = line.strip()
            # 跳过空行
            if not stripped:
                continue
            # 跳过注释
            if stripped.startswith('#'):
                continue
            # 跳过以特定模式开头的行
            if any(stripped.startswith(p) for p in skip_start):
                continue
            # 跳过只有return的行
            if stripped.startswith('return') and len(stripped.split()) < 2:
                continue
            # 保留有逻辑的行（包含for/while/if或赋值）
            if '=' in stripped or 'for ' in stripped or 'while ' in stripped or 'if ' in stripped:
                blankable.append((idx, line, stripped))

        if len(blankable) < 2:
            return code, []

        num_blanks = min(num_blanks, len(blankable))
        selected = random.sample(blankable, num_blanks)
        blank_indices = {s[0] for s in selected}

        blanks = []
        blanked_code = []

        for idx, line in enumerate(lines):
            if idx in blank_indices:
                for _, _, stripped in blankable:
                    if stripped in line:
                        # 生成挖空提示
                        if 'for ' in stripped or 'while ' in stripped:
                            blanks.append(("填写循环条件", stripped))
                            blanked_code.append('    # 请填写循环条件')
                        elif 'if ' in stripped:
                            blanks.append(("填写判断条件", stripped))
                            blanked_code.append('    # 请填写判断条件')
                        elif ' = ' in stripped and '==' not in stripped:
                            blanks.append(("填写赋值表达式", stripped))
                            blanked_code.append('    # 请填写赋值表达式')
                        else:
                            blanks.append(("填写此行", stripped))
                            blanked_code.append('    # 请填写')
                        break
            else:
                blanked_code.append(line)

        return '\n'.join(blanked_code), blanks

    def get_pitfalls(self, question):
        """根据题目动态生成易错点"""
        name = question.get('name', '').lower()
        number = question.get('number', '')

        # 默认易错点
        default = [
            "边界条件检查（空列表、单个元素等）",
            "循环条件的边界处理",
            "索引越界问题"
        ]

        # 按题型分类
        if any(k in name for k in ['链表', 'linked', 'list']):
            return [
                "空链表检查",
                "指针丢失问题",
                "循环终止条件",
                "dummy节点的使用"
            ]
        elif any(k in name for k in ['树', 'tree']):
            return [
                "空树检查",
                "递归终止条件",
                "遍历顺序（先序/中序/后序）",
                "层序遍历的队列使用"
            ]
        elif any(k in name for k in ['动态规划', 'dp', '爬楼梯', '斐波那契']):
            return [
                "dp数组初始化",
                "状态转移方程",
                "遍历顺序",
                "空间优化可能性"
            ]
        elif any(k in name for k in ['排序', 'sort', '快排', '归并']):
            return [
                "基准值选择",
                "分区边界处理",
                "递归深度",
                "稳定性"
            ]
        elif any(k in name for k in ['回溯', 'backtrack', '排列', '组合']):
            return [
                "选择列表管理",
                "撤销选择",
                "终止条件",
                "去重逻辑"
            ]
        elif any(k in name for k in ['哈希', 'hash', 'map']):
            return [
                "键值对映射",
                "重复处理",
                "空间复杂度"
            ]
        elif any(k in name for k in ['双指针', 'two pointer', '滑动窗口']):
            return [
                "指针移动条件",
                "窗口大小控制",
                "边界处理"
            ]
        elif any(k in name for k in ['二分', 'binary', '搜索']):
            return [
                "搜索区间定义",
                "中位数计算",
                "左右边界选择"
            ]
        elif any(k in name for k in ['路径', 'path', '最短路径']):
            return [
                "dp数组初始化",
                "状态转移方向",
                "边界条件处理",
                "空间优化"
            ]
        elif any(k in name for k in ['子集', 'subset', '组合', '全排列']):
            return [
                "选择列表管理",
                "递归深度控制",
                "去重处理",
                "终止条件"
            ]
        elif any(k in name for k in ['最大', '最小', '最长', '最短']):
            return [
                "状态定义",
                "状态转移方程",
                "初始化边界",
                "结果收集"
            ]
        else:
            return default

    def print_separator(self, char='=', length=50):
        print('\n' + char * length + '\n')

    def show_leetcode_explanation(self, question):
        self.print_separator('=')
        print(f"[B] LeetCode #{question['number']}: {question['name']}")
        print(f"[L] 链接: {question['link']}")
        self.print_separator('-')

        print("【题目描述】")
        print(question['description'])
        self.print_separator('-')

        print("【核心思路】")
        print(question['thought'])
        self.print_separator('-')

        print("【标准代码】")
        print("```python")
        print(question['code'])
        print("```")

        # 根据题目类型生成易错点
        pitfalls = self.get_pitfalls(question)
        print("[!] 容易卡住的地方:")
        for p in pitfalls:
            print(f"- {p}")

        print("\n[D] 调试技巧:")
        print("1. 用print打印中间变量")
        print("2. 用小数据测试逻辑")
        print("3. 画图辅助理解")

    def practice_fill_blanks(self, question):
        self.print_separator()
        print("[P] 【挖空练习】请填写以下代码")

        blanked_code, blanks = self.create_fill_blanks(question['code'], num_blanks=2)

        print("\n【挖空版本】")
        print("```python")
        print(blanked_code)
        print("```")

        print("\n" + "─" * 40)
        print("【第1次填空】")
        print(f"请填写: {blanks[0][0]}")
        input("按回车查看答案: ")
        print(f"[OK] 答案: {blanks[0][1]}")

        if len(blanks) > 1:
            print("\n" + "─" * 40)
            print("【第2次填空】")
            print(f"请填写: {blanks[1][0]}")
            input("按回车查看答案: ")
            print(f"[OK] 答案: {blanks[1][1]}")

    def practice_full_code(self, question):
        self.print_separator()
        print("[W] 【完整练习】请自己写出完整代码")
        print(f"\n题目: {question['name']}")
        print("提示: 先理解思路，然后自己写一遍！")

        user_code = []
        print("\n请输入你的代码（输入'完成'结束，输入'跳过'放弃）:")
        while True:
            line = input(">>> ")
            if line == "完成":
                break
            if line == "跳过":
                print("已跳过本次练习")
                return
            user_code.append(line)

        if user_code:
            print("\n【你写的代码】")
            print('\n'.join(user_code))

            print("\n【参考代码】")
            print("```python")
            print(question['code'])
            print("```")

    def show_llm_topic(self, topic):
        self.print_separator('=')
        print(f"[R] LLM知识点: {topic['title']}")
        print(f"分类: {topic['category']}")
        self.print_separator('-')

        print("【知识点讲解】")
        print(topic['content'])

    def practice_llm_fill_blank(self, topic):
        self.print_separator()
        print("[P] 【LLM知识点挖空练习】")

        if topic['blanks']:
            blank = random.choice(topic['blanks'])
            print(f"\n请填空: {blank[0]}")
            input("按回车查看答案: ")
            print(f"[OK] 答案: {blank[1]}")
        else:
            print("本知识点暂无挖空题")

    def record_to_file(self, questions, llm_topic):
        """记录到文档"""
        today = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 记录LeetCode题目
        record_lines = [f"\n## 练习日期: {today}\n"]

        for q in questions:
            record_lines.append(f"### LeetCode #{q['number']}: {q['name']}")
            record_lines.append(f"链接: {q['link']}")
            record_lines.append(f"")
            record_lines.append(f"**核心思路**: {q['thought'][:200]}...")
            record_lines.append(f"")
            record_lines.append(f"**代码**:")
            record_lines.append("```python")
            record_lines.append(q['code'])
            record_lines.append("```")
            record_lines.append("")

        # 记录LLM知识点
        record_lines.append(f"### LLM知识点: {llm_topic['title']}")
        record_lines.append(f"分类: {llm_topic['category']}")
        record_lines.append(f"")
        record_lines.append(f"**内容**: {llm_topic['content'][:300]}...")
        record_lines.append("")

        # 检查是否重复，不重复则追加
        new_content = "\n".join(record_lines)

        if os.path.exists(RECORD_FILE):
            with open(RECORD_FILE, 'r', encoding='utf-8') as f:
                existing = f.read()

            # 检查是否已存在（简单检查日期和第一道题）
            if today.split()[0] in existing:
                print(f"\n[*] 今日记录已存在，跳过写入")
                return
        else:
            existing = "# 练习记录\n"

        with open(RECORD_FILE, 'w', encoding='utf-8') as f:
            f.write(existing + new_content)

        print(f"[*] 已记录到 {RECORD_FILE}")

    def run(self):
        print("\n" + "**" * 15)
        print("**  每日算法+AI手撕练习 v2.0  **")
        print("**  新题+旧题混合练习  **")
        print("**" * 15)

        today = datetime.now().strftime("%Y-%m-%d")

        # 每天都可以练习，不再限制
        print(f"\n[*] 欢迎回来练习！")

        # 选择题目
        questions = self.select_questions(2)
        if not questions:
            print("[X] 没有找到题目文件")
            return

        # 显示新旧题比例
        history = self.progress.get('leetcode_history', [])
        new_count = sum(1 for q in questions if q['id'] not in history)
        print(f"[*] 本次: {len(questions)}道题 (新题{new_count}道, 旧题{len(questions)-new_count}道)")

        # 练习LeetCode
        for i, question in enumerate(questions):
            print(f"\n\n{'='*60}")
            is_old = question['id'] in history
            mark = "[复习]" if is_old else "[新题]"
            print(f"[=] 练习 {i+1}/2: {mark} LeetCode #{question['number']} {question['name']}")
            print(f"{'='*60}")

            self.show_leetcode_explanation(question)
            input("\n[R] 看完讲解后，按回车开始挖空练习...")
            self.practice_fill_blanks(question)
            input("\n[=] 按回车开始完整练习...")
            self.practice_full_code(question)

            # 记录进度（累加，不删除）
            if question['id'] not in self.progress['leetcode_history']:
                self.progress['leetcode_history'].append(question['id'])

            # 练习次数+1
            counts = self.progress.get('leetcode_counts', {})
            counts[question['id']] = counts.get(question['id'], 0) + 1
            self.progress['leetcode_counts'] = counts

        # 练习LLM
        print("\n\n" + ">>" * 15)
        print(">> 接下来是LLM知识点时间！")
        print(">>" * 15)

        llm_topic = self.select_llm_topic()
        self.show_llm_topic(llm_topic)
        input("\n[R] 看完知识点后，按回车开始挖空练习...")
        self.practice_llm_fill_blank(llm_topic)

        # 记录LLM进度
        if llm_topic['id'] not in self.progress['llm_history']:
            self.progress['llm_history'].append(llm_topic['id'])

        llm_counts = self.progress.get('llm_counts', {})
        llm_counts[llm_topic['id']] = llm_counts.get(llm_topic['id'], 0) + 1
        self.progress['llm_counts'] = llm_counts

        self.progress['total_days'] = self.progress.get('total_days', 0) + 1
        self._save_progress()

        # 记录到文档
        self.record_to_file(questions, llm_topic)

        self.print_separator('==')
        print("== 今日练习完成！")
        print(f"[S] LeetCode累计: {len(self.progress['leetcode_history'])}道")
        print(f"[S] LLM知识点累计: {len(self.progress['llm_history'])}个")
        print(f"[S] 总练习天数: {self.progress['total_days']}天")
        print("==" * 10)


def main():
    app = DailyPracticeApp()
    app.run()


if __name__ == "__main__":
    main()
