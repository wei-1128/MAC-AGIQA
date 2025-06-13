import numpy as np
from collections import defaultdict
import requests

class GlobalScorer:
    def __init__(self, api_key,lambda_penalty=0.5, model="glm-4v-flash"):
        self.lambda_penalty = lambda_penalty
        self.api_key = api_key
        self.original_prompt = None
        self.optimized_prompt = None
        self.quality_summary = None
        self.model = model
        assert isinstance(lambda_penalty, (int, float)), \
            f"lambda_penalty 必须是数值，实际是 {type(lambda_penalty)}"
        self.lambda_penalty = float(lambda_penalty)  # 强制转换
        
    def compute_final_score(self, mos_cq, mos_tc, explanations_cq, explanations_tc, prompt_text):
        # 类型安全检查
        print(mos_cq,mos_tc)
        # 计算基础分
        base_score = 0.5 * (mos_cq + mos_tc)
        print("111")
        # 计算惩罚项（内部已做类型检查）
        penalty = self._calculate_penalty(mos_cq, mos_tc)
        print("222")
        # 最终评分
        self.lambda_penalty * penalty
        print("333")
        final_score = np.clip(base_score + self.lambda_penalty * penalty, 0, 1)
        print("444")
        # 生成反馈
        feedback = self._generate_feedback(explanations_cq, explanations_tc, mos_cq, mos_tc, prompt_text)
        
        return final_score, feedback

    def _calculate_penalty(self, mos_cq, mos_tc):
        """根据论文表1计算惩罚项"""
        # 高质量但意图未表达 (CQ高，TC低)
        if mos_cq >= 0.7 and mos_tc <= 0.4:
            return -0.1
        
        # 符合意图但质量差 (TC高，CQ低)
        elif mos_tc >= 0.7 and mos_cq <= 0.4:
            return -0.05
        
        # 严重失真且不符语义 (双低)
        elif mos_cq <= 0.4 and mos_tc <= 0.4:
            return -0.3
        
        # 高质量图像不惩罚
        else:
            print("555")
            return 0
    def _generate_feedback(self, explanations_cq, explanations_tc, mos_cq, mos_tc, prompt_text):
        # 构建大模型提示
        analysis_prompt = f"""你是一个专业的AI生成质量分析师，请根据以下评估数据生成质量总结和优化后的提示词，优化后的提示词一定是为了解决目前的问题而生的：
                            [原始提示词]
                            {prompt_text}
                            [内容质量评估]（得分：{mos_cq:.2f}/1.0）
                            {str(explanations_cq)}
                            [图文一致性评估]（得分：{mos_tc:.2f}/1.0）
                            {str(explanations_tc)}
                            请按以下格式输出：
                            ### 质量总结（短摘要） ###
                            ### 优化后的提示词 ###
                        """
        try:
            print(111)
            print(str(explanations_cq))
            # 调用大模型API
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.post(
                "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{
                        "role": "user", 
                        "content": analysis_prompt
                    }]
                },
                headers=headers,
                timeout=15
            )
            return response.json()["choices"][0]["message"]["content"]

        except Exception as e:
            return f"""### 质量总结 ###
                        内容质量评分：{mos_cq:.2f}
                        图文一致评分：{mos_tc:.2f}
                        ### 优化提示 ###
                        {prompt_text}, enhanced quality"""


if __name__ == "__main__":
    # 模拟输入数据
    mos_cq = 1.0  # 假设图像质量较差
    mos_tc = 1.0  # 图文一致性较好
    prompt_text = "A giraffe walking through a green grass covered field"

    # 模拟三个智能体、两轮的解释数据（每轮每个问题一个解释）
    explanations_cq = [
        {0: ["0 + 长颈鹿腿部比例不协调", "0 + 草地纹理重复", "1 + 图像边缘清晰"]},
        {0: ["0 + 结构不自然", "0 + 草地颜色分布不均", "1 + 细节表现良好"]},
        {0: ["0 + 长颈鹿头身比例异常", "0 + 背景有混合伪影", "1 + 没有模糊"]},
    ]

    explanations_tc = [
        {0: ["1 + 图像中有长颈鹿", "1 + 草地绿色明显", "1 + 符合“行走”语义"]},
        {0: ["1 + 物体与提示一致", "1 + 背景场景正确", "1 + 动作表达清晰"]},
        {0: ["1 + 存在指定动物", "1 + 绿色草地可辨", "1 + 动作与提示匹配"]},
    ]

    # 初始化 scorer
    scorer = GlobalScorer(api_key="74a5a65105194797a4613f0b921266ba.N1y1c5frioiFJFvA")

    # 运行评分与反馈
    score, feedback = scorer.compute_final_score(
        mos_cq=mos_cq,
        mos_tc=mos_tc,
        explanations_cq=explanations_cq,
        explanations_tc=explanations_tc,
        prompt_text=prompt_text
    )

    print("🟢 最终评分：", score)
    print("\n📋 反馈建议：")
    print(feedback)