from question_gen import QuestionGenerator
from agents import AgentEvaluator
from fusion import WeightedFusion
from scoring import GlobalScorer
from utils import load_image_as_base64, init_logger

class MACAGIQA:
    def __init__(self, api_key, model="glm-4v-flash"):
        self.api_key = api_key
        self.model = model
        self.logger = init_logger("MAC-AGIQA")
        # 初始化各子模块
        self.qg = QuestionGenerator(api_key, model)
        self.agent_eval = AgentEvaluator(api_key, model)
        self.fuser = WeightedFusion()
        self.scorer = GlobalScorer(api_key,0.5,model)

    def run(self, image_path, prompt_text):
        self.logger.info("🔄 加载图像并编码为Base64...")
        try:
            image_base64 = load_image_as_base64(image_path)
        except Exception as e:
            self.logger.error(f"❌ 图像加载失败: {e}")
            return None, "图像加载失败"

        self.logger.info("🧠 Step 1: 正在生成内容质量与图文一致性问题...")
        try:
            q_cq, q_tc = self.qg.generate_questions(image_base64, prompt_text)
            print(q_cq,q_tc)
            self.logger.info(f"✅ 问题生成完成：CQ={len(q_cq)}，TC={len(q_tc)}")
        except Exception as e:
            self.logger.error(f"❌ 问题生成失败: {e}")
            return None, "问题生成失败"

        self.logger.info("👥 Step 2: 开始多智能体评估与多轮反思...")
        try:
            answers_cq ,explanations_cq= self.agent_eval.evaluate(image_base64, q_cq, module='CQ')
            answers_tc ,explanations_tc= self.agent_eval.evaluate(image_base64, q_tc, module='TC')
            print(answers_cq)
            self.logger.info("✅ 智能体回答与反思完成")
        except Exception as e:
            self.logger.error(f"❌ 智能体评估失败: {e}")
            return None, "智能体评估失败"

        self.logger.info("⚖️ Step 3: 加权融合各模块评分...")
        try:
            
            mos_cq = self.fuser.compute_module_score(answers_list=answers_cq,num_agents=3,num_questions=len(q_cq), num_rounds=3)
            mos_tc = self.fuser.compute_module_score(answers_list=answers_tc,num_agents=3,num_questions=len(q_tc), num_rounds=3)
            print(answers_cq)
            print(mos_cq,mos_tc)
            self.logger.info(f"📈 CQ评分={mos_cq:.4f}, TC评分={mos_tc:.4f}")
        except Exception as e:
            self.logger.error(f"❌ 模块评分融合失败: {e}")
            return None, "模块评分融合失败"

        self.logger.info("🌐 Step 4: 生成最终主观评分与反馈建议...")
        try:
            final_score, feedback = self.scorer.compute_final_score(mos_cq, mos_tc, explanations_cq, explanations_tc,prompt_text)
            self.logger.info(f"🏁 最终评分={final_score:.4f}")
        except Exception as e:
            self.logger.error(f"❌ 全局评分计算失败: {e}")
            return None, "全局评分失败"

        return final_score, feedback
if __name__ == "__main__":
    api_key = "74a5a65105194797a4613f0b921266ba.N1y1c5frioiFJFvA"
    image_path = "/Users/aonian/Documents/硕士/科研/多模态检测/论文实验/mac-agiqa/image/029-2.png"
    prompt = "An anime illustration of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blu"

    system = MACAGIQA(api_key)
    score, feedback = system.run(image_path, prompt)

    print("最终主观质量评分:", score)
    print("反馈：", feedback)
