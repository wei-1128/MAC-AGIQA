import requests
import time
import os
import json

class AgentEvaluator:
    def __init__(self, api_key, model="glm-4v-flash", num_agents=3, num_rounds=3):
        self.api_key = api_key
        self.model = model
        self.num_agents = num_agents
        self.num_rounds = num_rounds
        self.url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.agent_responses = [{} for _ in range(num_agents)]
        self.agent_explanations = [{} for _ in range(num_agents)]

    def evaluate(self, image_base64, question_list, module='CQ'):
        agents = [{} for _ in range(self.num_agents)]
        self.agent_responses = [{} for _ in range(self.num_agents)]
        self.agent_explanations = [{} for _ in range(self.num_agents)]

        for r in range(self.num_rounds):
            for i in range(self.num_agents):
                answers = []
                explanations = []
                for q_idx, q in enumerate(question_list):
                    prompt = self._construct_prompt(q, r, i, q_idx, module)
                    print(f"_____第{r+1}轮,第{i+1}个智能体-----------------")
                    response = self._query_model(prompt, image_base64)
                    answer, explanation = self._parse_answer(response)
                    answers.append(answer)
                    explanations.append(explanation)

                agents[i][r] = answers
                self.agent_responses[i][r] = {
                    'answers': answers,
                    'explanations': explanations
                }
                self.agent_explanations[i][r] = explanations
                time.sleep(0.3)
                self._save_to_json(module, r, question_list)
        return agents, self.agent_explanations

    def _construct_prompt(self, question, round_idx, agent_idx, q_idx, module):
        history = ""
        if round_idx > 0:
            for j in range(self.num_agents):
                if j != agent_idx and (round_idx - 1) in self.agent_responses[j]:
                    peer_answer = self.agent_responses[j][round_idx - 1]['answers'][q_idx]
                    peer_explanation = self.agent_responses[j][round_idx - 1]['explanations'][q_idx]
                    history += f"Agent{j+1} 的观点：评分 {peer_answer}，理由：{peer_explanation}\n"

        if module == 'CQ':
            scoring_rule = """
        【任务说明】
        你需要判断图像是否存在明显问题，返回如下格式：
        - `1 + 解释`：图像不存在提问中描述的问题。
        - `0.5 + 解释`：图像只是轻微存在提问中描述的问题。
        - `0 + 解释`：图像明显存在提问中描述的问题。

        特别说明：
        - 对于艺术化风格（如超现实主义），即便图像比例夸张、形态异常，只要符合艺术表达，也可给出较高评分。
        """

        elif module == 'TC':
            scoring_rule = """
        【评分规则 - 目标内容判断 (TC)】
        你需要判断图像是否表达了文本问题中提到的目标、属性、关系或场景，并返回以下格式：
        - `1 + 理由`：图像内容完全符合问题描述。
        - `0.5 + 理由`：图像内容部分符合，存在模糊或遗漏之处。
        - `0 + 理由`：图像内容不符合问题描述。
        """
        else:
            scoring_rule = "【评分规则未定义】"
        
        prompt = f"""
你是一个多轮AI图像评估智能体，善于分析图像质量或图像理解能力。
请根据下列图像以及问题进行判断，并返回以下格式：
`1 + 解释` 或 `0.5 + 解释` 或 `0 + 解释`
【问题】
{question}
{scoring_rule}
"""
        if round_idx > 0:
            prompt += f"\n【其他智能体历史观点】，\n{history},\n请结合其他智能体观点并反思。"
        return prompt

    def _query_model(self, prompt, image_base64):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        print(prompt)
        data = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]}
            ]
        }
        response = requests.post(self.url, headers=headers, json=data)
        print(response.json()["choices"][0]["message"]["content"])
        return response.json()["choices"][0]["message"]["content"]

    def _parse_answer(self, response_text):
        import re
        clean_text = re.sub(r"[`'\"]", "", response_text.strip())

        # 支持0, 1, 和0.5（或写作 0.50）
        match = re.match(r"^(1|0\.5|0)\s*\+\s*(.*)", clean_text)
        if match:
            score_str = match.group(1)
            explanation = match.group(2).strip()
            return float(score_str), explanation

        # fallback
        return 0.0, clean_text

    def _save_to_json(self, module, round_idx, question_list):
        save_data = []
        for i, agent_round in enumerate(self.agent_responses):
            if round_idx in agent_round:
                agent_entry = {
                    "agent_id": f"Agent{i+1}",
                    "round": round_idx + 1,
                    "module": module,
                    "responses": []
                }
                answers = agent_round[round_idx]['answers']
                explanations = agent_round[round_idx]['explanations']
                for idx, (ans, expl) in enumerate(zip(answers, explanations)):
                    agent_entry["responses"].append({
                        "question_index": idx + 1,
                        "question_text": question_list[idx],
                        "answer": ans,
                        "explanation": expl
                    })
                save_data.append(agent_entry)

        os.makedirs("logs", exist_ok=True)
        filename = f"logs/{module}_round_{round_idx+1}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=4)

