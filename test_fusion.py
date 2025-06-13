
from question_gen import QuestionGenerator
from agents import AgentEvaluator
from fusion import WeightedFusion
from scoring import GlobalScorer
from utils import load_image_as_base64, init_logger

answers_cq = [
    {0: [0, 1, 0, 1, 1, 0], 1: [0, 0, 0, 1, 0, 0], 2: [0, 0, 0, 1, 0, 0]},
    {0: [0, 0, 0, 1, 1, 0], 1: [0, 0, 0, 1, 0, 0], 2: [0, 0, 0, 1, 0, 0]},
    {0: [0, 1, 0, 1, 1, 1], 1: [0, 1, 0, 1, 0, 0], 2: [0, 0, 0, 1, 0, 0]}
]

answers_tc = [
    {0: [0, 0, 0, 1, 1, 0], 1: [0, 0, 0, 1, 0, 0], 2: [0, 0, 0, 1, 0, 0]},
    {0: [0, 0, 0, 1, 1, 0], 1: [0, 0, 0, 1, 0, 0], 2: [0, 0, 0, 1, 0, 0]},
    {0: [0, 1, 0, 1, 1, 1], 1: [0, 1, 0, 1, 0, 0], 2: [0, 0, 0, 1, 0, 0]}
]

mos_cq = WeightedFusion.compute_module_score(answers_list=answers_cq,num_agents=3,num_questions=6, num_rounds=3)
mos_tc = WeightedFusion.compute_module_score(answers_list=answers_tc,num_agents=3,num_questions=6, num_rounds=3)
print(mos_cq,mos_tc)