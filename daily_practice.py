# -*- coding: utf-8 -*-
"""
每日算法+AI手撕练习脚本
功能：
1. 每日推送两道LeetCode Hot100题目
2. 每日推送一个大模型知识点
3. 交互式练习：详细讲解 -> 挖空填空 -> 完整练习
"""

import os
import re
import random
import json
import sys
from datetime import datetime

# 设置UTF-8输出，解决Windows控制台编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 简单的emoji映射（兼容Windows）
EMOJI_MAP = {
    'target': '[ ]',
    'check': '[v]',
    'book': '[B]',
    'link': '[L]',
    'star': '*',
    'fire': '**',
    'rocket': '>>',
    'pen': '::',
    'memo': '==',
    'tada': '++'
}

# ============ 配置 ============
DATA_DIR = "D:/2026_project"
LEETCODE_EASY_FILE = os.path.join(DATA_DIR, "Hot100_Easy.md")
LEETCODE_MEDIUM_FILE = os.path.join(DATA_DIR, "Hot100_Medium.md")
PROGRESS_FILE = os.path.join(DATA_DIR, "daily_progress.json")
LLM_KNOWLEDGE_FILE = os.path.join(DATA_DIR, "llm_knowledge.json")

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
    ]
}


class DailyPracticeApp:
    def __init__(self):
        self.questions = []
        self.current_index = 0
        self.llm_topics = self._flatten_llm_topics()
        self.progress = self._load_progress()

    def _flatten_llm_topics(self):
        """将LLM知识点展平为列表"""
        topics = []
        for category, items in LLM_KNOWLEDGE.items():
            for item in items:
                topics.append({
                    'category': category,
                    'title': item['title'],
                    'content': item['content'],
                    'blanks': item['blanks']
                })
        return topics

    def _load_progress(self):
        """加载学习进度"""
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'leetcode_done': [],
            'llm_done': [],
            'last_date': None
        }

    def _save_progress(self):
        """保存学习进度"""
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, ensure_ascii=False, indent=2)

    def parse_leetcode_file(self, filepath):
        """解析LeetCode题目文件"""
        if not os.path.exists(filepath):
            print(f"文件不存在: {filepath}")
            return []

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        questions = []
        # 按###分割每个题目
        parts = re.split(r'(?=### \[)', content)

        for part in parts:
            if not part.strip() or not part.startswith('###'):
                continue

            # 提取题目编号和名称
            title_match = re.search(r'###\s*\[(\d+)\.\s*(.+?)\]', part)
            if not title_match:
                continue

            number = title_match.group(1)
            name = title_match.group(2)

            # 提取题目描述
            desc_match = re.search(r'#### 题目描述\s*(.+?)(?=####|$)', part, re.DOTALL)
            description = desc_match.group(1).strip() if desc_match else ""

            # 提取核心思路
            thought_match = re.search(r'#### 核心思路\s*(.+?)(?=####|$)', part, re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else ""

            # 提取代码
            code_match = re.search(r'```python\s*(.+?)```', part, re.DOTALL)
            code = code_match.group(1).strip() if code_match else ""

            # 提取链接
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

    def load_questions(self):
        """加载所有LeetCode题目"""
        self.questions = []
        self.questions.extend(self.parse_leetcode_file(LEETCODE_EASY_FILE))
        self.questions.extend(self.parse_leetcode_file(LEETCODE_MEDIUM_FILE))

        # 按题目编号排序
        self.questions.sort(key=lambda x: int(x['number']))

        # 过滤掉已完成的题目
        available = [q for q in self.questions if q['id'] not in self.progress['leetcode_done']]

        # 随机选择2道
        if len(available) >= 2:
            return random.sample(available, 2)
        elif available:
            return available
        else:
            # 如果都做完了，随机返回2道
            return random.sample(self.questions, 2)

    def get_llm_topic(self):
        """获取一个LLM知识点"""
        available = [t for t in self.llm_topics if t['title'] not in self.progress['llm_done']]

        if available:
            topic = random.choice(available)
        else:
            # 如果都学完了，重新随机
            topic = random.choice(self.llm_topics)

        return topic

    def create_fill_blanks(self, code, num_blanks=2):
        """创建挖空版本的代码"""
        # 提取有意义的代码行（排除空行和注释行）
        lines = code.split('\n')
        code_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('"""'):
                # 给每行加行号，方便练习
                code_lines.append(line)

        if len(code_lines) < 3:
            # 代码太短，返回原代码
            return code, []

        # 随机选择要挖空的行
        num_blanks = min(num_blanks, len(code_lines) - 1)
        blank_indices = random.sample(range(len(code_lines)), num_blanks)
        blank_indices.sort()

        # 创建挖空版本和答案
        blanks = []
        blanked_code = []

        for i, line in enumerate(code_lines):
            if i in blank_indices:
                # 提取该行的关键部分
                stripped = line.strip()
                # 尝试提取赋值语句的右侧或函数调用的关键部分
                if '=' in stripped and '==' not in stripped:
                    parts = stripped.split('=', 1)
                    if len(parts) == 2 and parts[1].strip():
                        blanks.append((parts[0].strip() + ' = ?', parts[1].strip()))
                        blanked_code.append('    # 请填写这一行: ' + parts[0].strip())
                        continue
                # 对于其他情况，整行挖空
                blanks.append((line.strip(), line.strip()))
                blanked_code.append('    # 请填写这一行')
            else:
                blanked_code.append(line)

        return '\n'.join(blanked_code), blanks

    def print_separator(self, char='=', length=60):
        print('\n' + char * length + '\n')

    def show_leetcode_explanation(self, question):
        """展示题目详解"""
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

        # 常见问题和调试技巧
        print("【[!] 容易卡住的地方】")
        print("1. 边界条件检查（空列表、单个元素等）")
        print("2. 循环条件的边界")
        print("3. 索引越界问题")

        print("\n【[D] 调试技巧】")
        print("1. 用print打印中间变量")
        print("2. 用小数据测试逻辑")
        print("3. 画图辅助理解")

    def practice_fill_blanks(self, question):
        """挖空练习"""
        self.print_separator()
        print("[P]  【挖空练习】请填写以下代码")

        blanked_code, blanks = self.create_fill_blanks(question['code'], num_blanks=2)

        print("\n【挖空版本】")
        print("```python")
        print(blanked_code)
        print("```")

        # 第一次填空
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
        """完整练习"""
        self.print_separator()
        print("[W]  【完整练习】请自己写出完整代码")
        print(f"\n题目: {question['name']}")
        print("提示: 先理解思路，然后自己写一遍！")
        print("\n开始计时...")

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
        """展示LLM知识点"""
        self.print_separator('=')
        print(f"[R] LLM知识点: {topic['title']}")
        print(f"📁 分类: {topic['category']}")
        self.print_separator('-')

        print("【知识点讲解】")
        print(topic['content'])

    def practice_llm_fill_blank(self, topic):
        """LLM知识点挖空"""
        self.print_separator()
        print("[P]  【LLM知识点挖空练习】")

        if topic['blanks']:
            blank = random.choice(topic['blanks'])
            print(f"\n请填空: {blank[0]}")
            input("按回车查看答案: ")
            print(f"[OK] 答案: {blank[1]}")
        else:
            print("本知识点暂无挖空题")

    def run(self):
        """运行每日练习"""
        print("\n" + "**" * 15)
        print("**  每日算法+AI手撕练习  **")
        print("**" * 15)

        today = datetime.now().strftime("%Y-%m-%d")

        # 检查今天是否已经练习过
        if self.progress.get('last_date') == today and len(self.progress['leetcode_done']) > 0:
            print(f"\n[!]  你今天已经练习过了！")
            choice = input("要重新开始吗？(y/n): ")
            if choice.lower() != 'y':
                return

        # 加载题目
        questions = self.load_questions()
        if not questions:
            print("[X] 没有找到题目文件")
            return

        print(f"\n[OK] 加载了 {len(questions)} 道题目")

        # 练习LeetCode题目
        for i, question in enumerate(questions):
            print(f"\n\n{'='*60}")
            print(f"[=] 练习 {i+1}/2: LeetCode #{question['number']} {question['name']}")
            print(f"{'='*60}")

            # 1. 详细讲解
            self.show_leetcode_explanation(question)

            # 2. 挖空练习
            input("\n[R] 看完讲解后，按回车开始挖空练习...")
            self.practice_fill_blanks(question)

            # 3. 完整练习
            input("\n[=] 按回车开始完整练习...")
            self.practice_full_code(question)

            # 标记完成
            self.progress['leetcode_done'].append(question['id'])
            self._save_progress()

        # 练习LLM知识点
        print("\n\n" + ">>" * 15)
        print(">> 接下来是LLM知识点时间！")
        print(">>" * 15)

        llm_topic = self.get_llm_topic()
        self.show_llm_topic(llm_topic)

        input("\n[R] 看完知识点后，按回车开始挖空练习...")
        self.practice_llm_fill_blank(llm_topic)

        # 标记完成
        self.progress['llm_done'].append(llm_topic['title'])
        self.progress['last_date'] = today
        self._save_progress()

        self.print_separator('==')
        print("== 今日练习完成！")
        print(f"[S] LeetCode进度: {len(self.progress['leetcode_done'])}/{len(self.questions)}")
        print(f"[S] LLM知识点进度: {len(self.progress['llm_done'])}/{len(self.llm_topics)}")
        print("==" * 10)


def main():
    app = DailyPracticeApp()
    app.run()


if __name__ == "__main__":
    main()
