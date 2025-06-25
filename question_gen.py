import requests
import json
import os
from datetime import datetime

class QuestionGenerator:
    def __init__(self, api_key, model="glm-4v-flash"):
        self.api_key = api_key
        self.model = model
        self.url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def generate_questions(self, image_base64, text_prompt, save_dir="generated_questions"):
        """
        生成问题并保存到JSON文件
        save_dir: 保存JSON文件的目录，默认为当前目录下的generated_questions
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成问题内容
        cq, tc = self._generate_questions(image_base64, text_prompt)
        tc_filtered = self._filter_questions_by_text_prompt(tc, text_prompt)
        # 构建保存数据结构
        data = {
            "metadata": {
                "model": self.model,
                "generated_time": datetime.now().isoformat(),
                "text_prompt": text_prompt
            },
            "content_quality": cq,
            "text_consistency": tc_filtered
        }
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"questions_{timestamp}.json"
        filepath = os.path.join(save_dir, filename)
        
        # 保存为JSON文件
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"问题已保存至：{filepath}")
        return cq, tc

    def _generate_questions(self, image_base64, text_prompt):
        """实际生成问题的逻辑"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 生成内容质量问题（CQ）
        cq_prompt = """你的任务是根据给定的 AI 生成图像，提出几个用于评估图像质量的具体“是/否型”问题。

请你参考以下 5 个维度提问：
1️⃣ 主体结构合理性（如肢体比例、面部结构、物体形态是否自然）【必须包含】
2️⃣ 细节真实性（纹理、材质、边缘是否真实或清晰）
3️⃣ 物理合理性（光影、透视、重力等是否自然）
4️⃣ 视觉伪影（是否存在模糊、噪点、鬼影、奇异边缘等）
5️⃣ AI 典型缺陷（是否有融合错误、重复纹理、结构崩坏等）

⚠️ 提问必须结合图像中的“**具体主体**”，不要泛泛提问。
⚠️ 所有问题必须以“是否……”开头，并可用“是/不是”来回答。
⚠️ 避免模糊措辞（如“是否自然”、“是否一致”等），要聚焦具体问题。

✅ 示例（以图像主体为“戴墨镜的男人”）：
1. 是否这个男人的双手结构正常？
2. 是否墨镜的边缘清晰无融合？
3. 是否光影在人物脸部和背景上分布合理？
4. 是否人物面部存在模糊或伪影？
5. 是否人物身体存在结构性错误或融合问题？

请以图像中的**主要人物或物体**为焦点，生成 5 条“是/否”问题，用来评估图像的质量。
"""

        cq_data = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": cq_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]}
            ]
        }
        cq_response = requests.post(self.url, headers=headers, json=cq_data)
        cq_content = cq_response.json()["choices"][0]["message"]["content"]
        cq_generated = self._parse_questions(cq_content)
                # 预设保底问题
        cq_fallback = [
            "是否图像存在模糊区域，特别是边缘和细节部分",
            "是否图像中存在噪声或颗粒感，特别是在阴影或低光区域",
            "是否图像中的纹理自然，是否有不一致的纹理或材质表面",
            "是否图像中存在明显的伪影或错误图像细节，特别是在边缘或复杂背景处",
            "是否图像中的物体排列是否自然，符合现实世界的空间关系",
            "是否图像中的物体布局有过度拥挤或过度空旷的区域，影响整体感知",
            "是否图像中的背景与主体内容协调，有视觉上的冲突或不匹配。",
            "是否图像中背景和主体在色彩、亮度、细节上相互协调，是否造成视觉的不和谐",
            "是否图像中的色彩真实自然，是否有不一致的色调或不自然的色彩变化。",
            "是否图像中存在过度饱和或色偏，特别是在阴影或高光区域。",
            "是否图像中的光照效果自然，是否存在过度曝光或曝光不足的情况",
            "是否图像中的高频细节是否因模糊而失真，图像是否看起来不够锐利？"
        ]

        # 如果模型生成的问题数量不足，则补足（最多保留5条）
        cq_final = (cq_generated + cq_fallback)
        # 生成图文一致性问题（TC）
        tc_prompt = f"""你是一个图文一致性测试问题生成助手，任务是根据以下 AI 图像生成提示词，提出 3~5 个用于判断图文是否一致的 **“是/否型问题”**。

【图像提示词】：
{text_prompt}

【问题设计要求】：
请围绕以下三类图文一致性进行提问：
1. **实体存在性判断**：提示词中提到的主要物体或场景是否真实出现在图像中？
2. **数量一致性判断**：提示词中如果明确了数量（如“一只马”或“两个人”），是否与图像一致？
3. **属性一致性判断**：提示词中提到的属性（如颜色、动作、姿态）是否在图像中体现？

⚠️ 提问必须严格基于提示词中**明确出现**的信息，不能编造不存在的细节。
⚠️ 如果提示词中没有提及颜色、数量、动作等，不允许自行添加提问。
⚠️ 禁止模糊提问或“是否正确”、“是否一致”、“是否与描述相符”等不明确表述。

【输出格式要求】：
- 每个问题必须是**“是否……”**句式，能用“是”或“不是”来回答。
- 所有问题以编号列出，格式如下：
1. xxx？
2. xxx？
...

请严格按照以上规范生成图文一致性问题。
"""
        print(tc_prompt)
        tc_data = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": tc_prompt}
                ]}
            ]
        }
        tc_response = requests.post(self.url, headers=headers, json=tc_data)
        tc_content = tc_response.json()["choices"][0]["message"]["content"]

        # 解析问题列表
        return cq_final , self._parse_questions(tc_content)

    def _parse_questions(self, content):
        """解析问题文本"""
        questions = []
        for line in content.split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                # 移除序号和点（支持1. 或 1、等形式）
                clean_line = line.split('.', 1)[-1].split('、', 1)[-1].strip()
                if clean_line:
                    questions.append(clean_line)
        return questions
    def _filter_questions_by_text_prompt(self, questions, text_prompt):
        """调用模型过滤掉与提示词无关或引入额外信息的问题"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        prompt = f"""你是一个内容审查助手。任务是检查以下“问题”是否忠实于图像提示词，没有引入提示词中未提到的额外信息。

    【图像提示词】：
    {text_prompt}

    【问题列表】：
    {chr(10).join([f"{i+1}. {q}" for i, q in enumerate(questions)])}

    【判断标准】：
    - ✅ 如果问题只涉及提示词中明确提到的实体、属性、场景、数量，保留；
    - ❌ 如果问题包含提示词中未出现的颜色、动作、姿态、数量、环境、角色等，则视为“不忠实”，剔除。

    请返回保留的问题列表，只保留“忠实”的问题。以编号 + 问题形式输出，每行一个问题。
    """

        data = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        }

        response = requests.post(self.url, headers=headers, json=data)
        content = response.json()["choices"][0]["message"]["content"]

        # 解析返回的保留问题
        return self._parse_questions(content)