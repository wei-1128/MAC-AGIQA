# fusion.py

class WeightedFusion:
    def __init__(self, alpha=0.5):
        """
        alpha: 权重融合的平衡因子（用于平衡一致性与观点接受度）
        """
        self.alpha = alpha

    def compute_module_score(self, answers_list, num_agents, num_questions, num_rounds):
        """
        输入：
        - answers_list: List[Dict[int, List[int]]]，每轮的所有智能体回答
        - num_agents: int，智能体数量
        - num_questions: int，问题数量
        - num_rounds: int，反思轮次

        输出：
        - module_score: float，模块评分
        """
        print("开始计算模块评分...")
        print(f"输入数据：{answers_list}")
        print(f"参数：num_agents={num_agents}, num_questions={num_questions}, num_rounds={num_rounds}")

        # 初始化自一致性和观点接受度列表
        consistency_list = [0.0] * num_agents
        acceptance_list = [0.0] * num_agents

        # 计算自一致性
        print("计算自一致性...")
        for i in range(num_agents):
            inconsistent_count = 0
            for j in range(num_questions):
                for r in range(1, num_rounds):
                    # 注意这里比较的是同一个智能体在不同轮次的回答
                    if answers_list[i][r][j] != answers_list[i][r-1][j]:
                        inconsistent_count += 1
            consistency_list[i] = 1 - (inconsistent_count / (num_rounds - 1) / num_questions)
            print(f"智能体 {i} 的自一致性：{consistency_list[i]}")

        # 计算观点接受度
        print("计算观点接受度...")
        for i in range(num_agents):
            acceptance_count = 0
            for j in range(num_questions):
                for k in range(num_agents):
                    if k != i:
                        # 比较其他智能体最终轮回答与本智能体倒数第二轮的答案
                        if answers_list[k][num_rounds-1][j] == answers_list[i][num_rounds-2][j]:
                            acceptance_count += 1
            acceptance_list[i] = acceptance_count / ((num_agents - 1) * num_questions)
            print(f"智能体 {i} 的观点接受度：{acceptance_list[i]}")


        # 计算权重
        print("计算权重...")
        weights = [
            self.alpha * cons + (1 - self.alpha) * acc
            for cons, acc in zip(consistency_list, acceptance_list)
        ]
        print(f"权重列表：{weights}")

        # 计算每个问题的加权得分
        print("计算每个问题的加权得分...")
        fused_scores = []
        for j in range(num_questions):
            answers = [answers_list[-1][i][j] for i in range(num_agents)]
            fused_score = self.aggregate_scores(answers, weights)
            fused_scores.append(fused_score)
            print(f"问题 {j} 的加权得分：{fused_score}")

        # 计算模块评分
        print("计算模块评分...")
        module_score = sum(fused_scores) / len(fused_scores)
        print(f"模块评分：{module_score}")

        return module_score

    def aggregate_scores(self, answers, weights):
        """
        输入：
        - answers: List[int]，每个智能体对某个问题的回答（0 或 1）
        - weights: List[float]，对应的权重列表

        输出：
        - fused_score: float，加权融合结果
        """
        if not answers or not weights or sum(weights) == 0:
            return 0.0
        return sum(a * w for a, w in zip(answers, weights)) / sum(weights)
