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
        lines = code.split('\n')
        code_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('"""'):
                code_lines.append(line)

        if len(code_lines) < 3:
            return code, []

        num_blanks = min(num_blanks, len(code_lines) - 1)
        blank_indices = random.sample(range(len(code_lines)), num_blanks)
        blank_indices.sort()

        blanks = []
        blanked_code = []

        for i, line in enumerate(code_lines):
            if i in blank_indices:
                stripped = line.strip()
                if '=' in stripped and '==' not in stripped:
                    parts = stripped.split('=', 1)
                    if len(parts) == 2 and parts[1].strip():
                        blanks.append((parts[0].strip() + ' = ?', parts[1].strip()))
                        blanked_code.append('    # 请填写这一行: ' + parts[0].strip())
                        continue
                blanks.append((line.strip(), line.strip()))
                blanked_code.append('    # 请填写这一行')
            else:
                blanked_code.append(line)

        return '\n'.join(blanked_code), blanks

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

        print("[!] 容易卡住的地方:")
        print("1. 边界条件检查")
        print("2. 循环条件的边界")
        print("3. 索引越界问题")

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
