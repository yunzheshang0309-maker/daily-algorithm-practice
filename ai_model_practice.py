# -*- coding: utf-8 -*-
"""
AI模型手撕练习脚本
功能：
1. 步步深入的交互式AI模型实现练习
2. 大模型辅助提示和检查
3. 可查看答案，可重复练习
"""

import os
import re
import json
import sys
import requests
from datetime import datetime

# 设置UTF-8输出
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ============ 配置区 ============
DATA_DIR = "D:/2026_project"
PROGRESS_FILE = os.path.join(DATA_DIR, "ai_progress.json")

# MiniMax API配置（从daily_paper.py获取）
MINIMAX_API_KEY = "sk-cp-1kFbjMQf2tAsRZ5WagCyJYJUTx13Rwsx0lob7sNsYgid4ES4D6b1xTYqfb6zRSbd4pz0d4l95AavCfTnCDtgz3Ti2SvY2eef0_EcgI0v2IZdHipnoljZg0Q"
MINIMAX_API_URL = "https://api.minimaxi.com/anthropic/v1/messages"
MODEL_NAME = "MiniMax-M2.5"

# ============ AI手撕题目库 ============
AI_PRACTICE_TOPICS = [
    {
        "id": "gradient_descent",
        "title": "梯度下降法",
        "description": """实现批量梯度下降算法，用于优化一个简单的一元二次函数。

目标函数: f(x) = x^2 + 5x + 10
学习率: 0.1
初始值: x = -10
迭代次数: 20次

请一步步实现这个梯度下降算法。""",
        "steps": [
            {
                "name": "定义目标函数",
                "hint": "定义一个函数 f(x) = x**2 + 5*x + 10，返回函数值",
                "check": "检查是否定义了 f(x) 函数"
            },
            {
                "name": "定义梯度函数",
                "hint": "梯度是函数对x的导数。对于 f(x) = x^2 + 5x + 10，导数是 2x + 5",
                "check": "检查是否定义了梯度函数 grad(x)，返回值应为 2*x + 5"
            },
            {
                "name": "实现梯度下降循环",
                "hint": """使用公式: x = x - learning_rate * gradient
循环20次，每次更新x的值""",
                "check": "检查是否有梯度下降的迭代循环"
            },
            {
                "name": "运行并输出结果",
                "hint": "打印每次迭代后的x值和f(x)值，观察是否收敛到最小值",
                "check": "检查是否有打印输出的循环"
            }
        ],
        "answer": """import numpy as np

# 1. 定义目标函数 f(x) = x^2 + 5x + 10
def f(x):
    return x**2 + 5*x + 10

# 2. 定义梯度函数（导数）
def gradient(x):
    return 2*x + 5

# 3. 梯度下降
learning_rate = 0.1
x = -10  # 初始值
iterations = 20

print("梯度下降过程:")
for i in range(iterations):
    grad = gradient(x)
    x = x - learning_rate * grad
    print(f"迭代 {i+1}: x = {x:.4f}, f(x) = {f(x):.4f}")

print(f"\\n最终结果: x = {x:.4f}, 最小值约为 {f(x):.4f}")"""
    },
    {
        "id": "linear_regression",
        "title": "线性回归",
        "description": """使用最小二乘法实现简单线性回归。

给定数据:
X = [1, 2, 3, 4, 5]
y = [1.5, 3.8, 6.2, 8.9, 11.5]

目标: 找到 y = wx + b 中的 w 和 b

提示: 使用闭式解
w = sum((x-x_mean)*(y-y_mean)) / sum((x-x_mean)^2)
b = y_mean - w * x_mean""",
        "steps": [
            {
                "name": "准备数据",
                "hint": "定义X和y列表，计算均值x_mean和y_mean",
                "check": "检查是否有X和y数据，以及均值计算"
            },
            {
                "name": "计算w（斜率）",
                "hint": "w = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²\n先计算分子: sum((x-x_mean)*(y-y_mean))",
                "check": "检查是否实现了w的计算公式"
            },
            {
                "name": "计算b（截距）",
                "hint": "b = y_mean - w * x_mean",
                "check": "检查是否实现了b的计算"
            },
            {
                "name": "预测与评估",
                "hint": "用学到的w和b预测新数据 x=6，打印预测结果",
                "check": "检查是否有预测输出"
            }
        ],
        "answer": """import numpy as np

# 数据
X = np.array([1, 2, 3, 4, 5])
y = np.array([1.5, 3.8, 6.2, 8.9, 11.5])

# 计算均值
x_mean = np.mean(X)
y_mean = np.mean(y)

# 计算w (斜率)
numerator = np.sum((X - x_mean) * (y - y_mean))
denominator = np.sum((X - x_mean) ** 2)
w = numerator / denominator

# 计算b (截距)
b = y_mean - w * x_mean

print(f"线性回归结果: y = {w:.4f}x + {b:.4f}")

# 预测
x_new = 6
y_pred = w * x_new + b
print(f"当 x = {x_new} 时，预测 y = {y_pred:.4f}")

# 计算R²
y_pred_all = w * X + b
ss_res = np.sum((y - y_pred_all) ** 2)
ss_tot = np.sum((y - y_mean) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"R² = {r2:.4f}")"""
    },
    {
        "id": "logistic_regression",
        "title": "逻辑回归",
        "description": """实现二分类的逻辑回归算法。

数据: 肿瘤大小 vs 恶性/良性 (0/1)
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

使用sigmoid函数: σ(z) = 1 / (1 + e^(-z))
预测: p = σ(wx + b)
损失: 交叉熵损失""",
        "steps": [
            {
                "name": "定义sigmoid函数",
                "hint": "sigmoid(z) = 1 / (1 + exp(-z))",
                "check": "检查是否定义了sigmoid函数"
            },
            {
                "name": "初始化参数",
                "hint": "初始化 w=0, b=0，学习率 lr=0.1",
                "check": "检查是否有w, b, lr的初始化"
            },
            {
                "name": "实现梯度下降",
                "hint": """对每个样本:
- pred = sigmoid(w*x + b)
- error = y - pred
- w = w + lr * error * x
- b = b + lr * error
迭代100次""",
                "check": "检查是否有训练循环"
            },
            {
                "name": "预测",
                "hint": "使用训练好的模型预测新数据 x=5.5",
                "check": "检查是否有预测输出"
            }
        ],
        "answer": """import numpy as np

# 数据
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 初始化
w = 0
b = 0
lr = 0.1
iterations = 100

# 训练
for i in range(iterations):
    # 前向传播
    z = w * X + b
    predictions = sigmoid(z)

    # 计算梯度
    error = y - predictions
    dw = np.mean(error * X)
    db = np.mean(error)

    # 更新参数
    w += lr * dw
    b += lr * db

print(f"训练完成: w = {w:.4f}, b = {b:.4f}")

# 预测
x_test = 5.5
z = w * x_test + b
p = sigmoid(z)
print(f"x = {x_test}, 预测概率 = {p:.4f}, 预测类别 = {int(p > 0.5)}")"""
    },
    {
        "id": "kmeans",
        "title": "K-means聚类",
        "description": """实现K-means聚类算法。

数据点 (二维):
[[1, 2], [1, 4], [1, 0],
 [10, 2], [10, 4], [10, 0],
 [5, 5], [6, 5]]

k = 3 个簇

步骤:
1. 随机初始化3个中心点
2. 分配每个点到最近的中心
3. 更新中心点为簇的均值
4. 重复2-3步直到收敛""",
        "steps": [
            {
                "name": "准备数据和初始化中心",
                "hint": "定义数据点X，随机选择3个作为初始中心点 centroids",
                "check": "检查是否有数据定义和中心初始化"
            },
            {
                "name": "分配簇",
                "hint": """计算每个点到所有中心的距离
将点分配到距离最近的中心
可以用np.argmin找最小距离的索引""",
                "check": "检查是否有距离计算和簇分配"
            },
            {
                "name": "更新中心点",
                "hint": "对每个簇，计算所有点的均值作为新的中心",
                "check": "检查是否有中心更新逻辑"
            },
            {
                "name": "迭代直到收敛",
                "hint": "重复分配和更新，直到中心点不再变化（或达到最大迭代次数）",
                "check": "检查是否有迭代循环"
            }
        ],
        "answer": """import numpy as np

# 数据
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0],
              [5, 5], [6, 5]])

k = 3
max_iter = 20

# 初始化中心点 (选择前k个点作为初始中心)
centroids = X[:k].copy()

print("K-means聚类过程:")
for iteration in range(max_iter):
    # 分配簇: 计算每个点到每个中心的距离
    distances = np.zeros((len(X), k))
    for i in range(k):
        distances[:, i] = np.sqrt(np.sum((X - centroids[i]) ** 2, axis=1))

    # 找到最近的中心
    clusters = np.argmin(distances, axis=1)

    # 保存旧中心用于检查收敛
    old_centroids = centroids.copy()

    # 更新中心点
    for i in range(k):
        if np.any(clusters == i):
            centroids[i] = np.mean(X[clusters == i], axis=0)

    # 检查收敛
    if np.allclose(old_centroids, centroids):
        print(f"在第 {iteration + 1} 次迭代后收敛")
        break

    if iteration < 3:  # 只打印前几次
        print(f"迭代 {iteration + 1}: 中心 = {centroids}")

print(f"\\n最终聚类中心:\\n{centroids}")
print(f"\\n聚类结果: {clusters}")"""
    },
    {
        "id": "softmax",
        "title": "Softmax函数",
        "description": """实现Softmax函数，用于多分类。

给定 logits = [2.0, 1.0, 0.1]

Softmax公式: softmax(x_i) = exp(x_i) / sum(exp(x_j))

确保数值稳定性: 减去最大值
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))""",
        "steps": [
            {
                "name": "实现基础softmax",
                "hint": "对每个元素: exp(x_i) / sum(exp(x))",
                "check": "检查是否有softmax实现"
            },
            {
                "name": "添加数值稳定性",
                "hint": "先减去最大值: exp(x - max(x))，防止溢出",
                "check": "检查是否有数值稳定处理"
            },
            {
                "name": "验证结果",
                "hint": "验证softmax输出的总和为1",
                "check": "检查是否有验证输出"
            }
        ],
        "answer": """import numpy as np

def softmax(x):
    # 数值稳定性: 减去最大值
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)

# 测试
logits = np.array([2.0, 1.0, 0.1])
result = softmax(logits)

print("Softmax结果:")
print(f"输入: {logits}")
print(f"输出: {result}")
print(f"总和验证: {np.sum(result):.6f} (应为1.0)")

# 验证
expected = np.array([0.659, 0.242, 0.099])
print(f"预期值: {expected}")
print(f"接近预期: {np.allclose(result, expected, atol=0.01)}")"""
    },
    {
        "id": "neural_network",
        "title": "单层神经网络",
        "description": """实现一个最简单的神经网络：单层感知机。

实现 AND 逻辑门:
x1, x2 = 0/1
输出: x1 AND x2

网络结构:
输入: 2个节点
输出: 1个节点
激活: 阶跃函数 (x > 0 -> 1, else 0)""",
        "steps": [
            {
                "name": "准备AND门数据",
                "hint": "X = [[0,0], [0,1], [1,0], [1,1]], y = [0, 0, 0, 1]",
                "check": "检查是否有AND门数据"
            },
            {
                "name": "初始化参数",
                "hint": "随机初始化 w1, w2, b",
                "check": "检查是否有参数初始化"
            },
            {
                "name": "实现训练过程",
                "hint": """感知机学习规则:
- pred = step(w1*x1 + w2*x2 + b)
- error = y - pred
- w = w + lr * error * x
- b = b + lr * error""",
                "check": "检查是否有感知机训练逻辑"
            },
            {
                "name": "测试",
                "hint": "用训练好的网络预测所有4种输入",
                "check": "检查是否有预测测试"
            }
        ],
        "answer": """import numpy as np

# AND门数据
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# 阶跃函数
def step(x):
    return (x > 0).astype(int)

# 初始化
np.random.seed(42)
w = np.random.randn(2) * 0.5
b = 0
lr = 0.1
epochs = 20

# 训练
for epoch in range(epochs):
    for i in range(len(X)):
        # 前向
        z = np.dot(X[i], w) + b
        pred = step(z)

        # 更新
        error = y[i] - pred
        w += lr * error * X[i]
        b += lr * error

# 测试
print("单层感知机 AND门:")
print("输入\t\t输出\t预测")
for i in range(len(X)):
    z = np.dot(X[i], w) + b
    pred = step(z)
    print(f"{X[i]}\t{y[i]}\t{pred}")"""
    }
]


class AIPracticeApp:
    def __init__(self):
        self.current_topic = None
        self.current_step = 0
        self.user_code = ""
        self.progress = self._load_progress()

    def _load_progress(self):
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"topics_done": [], "times_practiced": {}}

    def _save_progress(self):
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, ensure_ascii=False, indent=2)

    def call_minimax(self, prompt, system_prompt="你是一个Python编程导师，擅长引导学生一步步实现算法。"):
        """调用MiniMax API"""
        payload = {
            "model": MODEL_NAME,
            "max_tokens": 1500,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}]
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": MINIMAX_API_KEY,
            "anthropic-version": "2023-06-01"
        }

        try:
            response = requests.post(MINIMAX_API_URL, headers=headers, json=payload, timeout=60)
            if response.status_code != 200:
                return f"API错误 ({response.status_code}): {response.text[:100]}"

            res_json = response.json()
            if 'content' in res_json:
                for block in res_json['content']:
                    if block.get('type') == 'text':
                        return block.get('text')
            return "未找到回复"
        except Exception as e:
            return f"调用失败: {str(e)}"

    def check_code_with_ai(self, user_code, step_info):
        """使用AI检查用户代码"""
        prompt = f"""用户正在实现一个AI模型，当前步骤是: {step_info['name']}
这个步骤的提示是: {step_info['hint']}

用户的代码:
```
{user_code}
```

请检查用户的代码是否正确实现这个步骤。如果有问题，指出具体问题并给出修正建议。如果看起来正确，请给出鼓励。"""

        return self.call_minimax(prompt)

    def get_hint_from_ai(self, topic, step):
        """使用AI生成更详细的提示"""
        prompt = f"""用户正在实现 {topic['title']}
当前步骤 {step+1}/{len(topic['steps'])}: {topic['steps'][step]['name']}

请给出更详细的编程指导，帮助用户完成这个步骤。可以给出代码示例或解释。"""

        return self.call_minimax(prompt)

    def print_separator(self, char='-', length=50):
        print('\n' + char * length + '\n')

    def show_topics(self):
        """展示所有题目"""
        self.print_separator('=')
        print("       AI模型手撕练习 - 选题")
        self.print_separator('=')

        for i, topic in enumerate(AI_PRACTICE_TOPICS):
            status = "[DONE]" if topic['id'] in self.progress.get('topics_done', []) else ""
            times = self.progress.get('times_practiced', {}).get(topic['id'], 0)
            print(f"  {i+1}. {topic['title']} {status} (练习{times}次)")
        print()

    def select_topic(self):
        """选择题目"""
        self.show_topics()
        try:
            choice = int(input("请选择题目 (1-{}): ".format(len(AI_PRACTICE_TOPICS))))
            if 1 <= choice <= len(AI_PRACTICE_TOPICS):
                self.current_topic = AI_PRACTICE_TOPICS[choice - 1]
                self.current_step = 0
                return True
        except:
            pass
        print("无效选择")
        return False

    def show_current_task(self):
        """展示当前任务"""
        topic = self.current_topic
        self.print_separator('=')
        print(f"  题目: {topic['title']}")
        self.print_separator('-')
        print(topic['description'])
        self.print_separator()

    def show_current_step(self):
        """展示当前步骤"""
        step = self.current_topic['steps'][self.current_step]
        print(f"\n[步骤 {self.current_step + 1}/{len(self.current_topic['steps'])}]")
        print(f"任务: {step['name']}")
        print(f"\n提示: {step['hint']}")

    def get_user_code(self):
        """获取用户代码"""
        print("\n请输入你的代码（输入'完成'结束当前步骤，输入'跳过'放弃，输入'答案'查看完整答案）:")
        lines = []
        while True:
            line = input(">>> ")
            if line == "完成":
                break
            if line == "跳过":
                return "跳过"
            if line == "答案":
                return "答案"
            lines.append(line)
        return '\n'.join(lines)

    def practice_with_ai_help(self):
        """使用AI辅助的练习模式"""
        topic = self.current_topic
        print(f"\n开始练习: {topic['title']}")
        print("=" * 50)

        # 显示题目
        self.show_current_task()

        # 分步骤进行
        for step_idx in range(len(topic['steps'])):
            self.current_step = step_idx
            self.show_current_step()

            # 让用户输入
            code = self.get_user_code()

            if code == "答案":
                print("\n[参考答案]")
                print("```python")
                print(topic['answer'])
                print("```")
                input("\n按回车继续...")
                return

            if code == "跳过":
                continue

            if code.strip():
                # 调用AI检查
                print("\n[AI检查中...]")
                feedback = self.check_code_with_ai(code, topic['steps'][step_idx])
                print("\n[AI反馈]")
                print(feedback[:500])

                # 让用户选择下一步
                print("\n请选择:")
                print("  1. 继续下一步")
                print("  2. 再试一次")
                print("  3. 查看详细提示")
                print("  4. 看答案")

                choice = input("你的选择 (1-4): ")
                if choice == "3":
                    print("\n[AI详细提示]")
                    hint = self.get_hint_from_ai(topic, step_idx)
                    print(hint[:500])
                elif choice == "4":
                    print("\n[参考答案]")
                    print("```python")
                    print(topic['answer'])
                    print("```")
                    return
                # 否则继续下一步或重试

        print("\n" + "=" * 50)
        print("练习完成!")
        print("=" * 50)

        # 记录进度
        if topic['id'] not in self.progress['topics_done']:
            self.progress['topics_done'].append(topic['id'])

        times = self.progress.get('times_practiced', {}).get(topic['id'], 0)
        self.progress.setdefault('times_practiced', {})[topic['id']] = times + 1
        self._save_progress()

    def show_answer(self):
        """展示答案"""
        print("\n" + "=" * 50)
        print("参考答案:")
        print("=" * 50)
        print("```python")
        print(self.current_topic['answer'])
        print("```")

    def run(self):
        """主循环"""
        print("\n" + "=" * 50)
        print("     AI模型手撕练习 v1.0")
        print("=" * 50)
        print("  梯度下降 | 线性回归 | 逻辑回归 | K-means")
        print("=" * 50)

        while True:
            if not self.select_topic():
                continue

            # 练习
            self.practice_with_ai_help()

            # 询问是否继续
            print("\n请选择:")
            print("  1. 再做一道")
            print("  2. 查看进度")
            print("  3. 退出")

            choice = input("你的选择: ")
            if choice == "3":
                break
            elif choice == "2":
                print(f"\n已完成: {len(self.progress.get('topics_done', []))} 个主题")
                total = sum(self.progress.get('times_practiced', {}).values())
                print(f"总练习次数: {total}")

        print("\n再见！坚持练习一定能进步！")


def main():
    app = AIPracticeApp()
    app.run()


if __name__ == "__main__":
    main()
