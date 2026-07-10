from methods.dlora import Learner as DLoraLearner

from .attention import Attention_LoRA
from .network import MANet
# 参考 FRLoRA，对 LoRA 做 SVD 主奇异空间初始化，然后每个 task 后做 residual accumulation
# LoRA initialization + accumulation
class Learner(DLoraLearner):
    network_cls = MANet
    attention_cls = Attention_LoRA
# 继承原来的 dLoRA 训练流程，
# 但是把 network 换成 ideas/lora_init_accum/network.py 里的 MANet，
# 把 attention 换成 ideas/lora_init_accum/attention.py 里的 Attention_LoRA。