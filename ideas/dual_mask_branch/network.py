import copy

import torch
import torch.nn as nn
from torch.nn import functional as F
try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_

from models.network import _create_vision_transformer
from .attention import Attention_LoRA


class MANet(nn.Module):
    def __init__(self, args):
        super(MANet, self).__init__()

        model_kwargs = dict(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            n_tasks=args["total_sessions"],
            rank=args["rank"],
            attn_fn=Attention_LoRA,
        )
        self.image_encoder = _create_vision_transformer(
            "vit_base_patch16_224_in21k",
            pretrained=True,
            **model_kwargs,
        )

        self.class_num = args["init_cls"]
        self.dim = args["embd_dim"]
        self.classifier_pool = nn.ModuleList(
            [
                nn.Linear(args["embd_dim"], self.class_num, bias=False)
                for _ in range(args["total_sessions"])
            ]
        )

        for m in self.classifier_pool.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)

        self.numtask = 0
        self.use_RP = False
        self.W_rand = None
        self.weight = None
        # self.eval_logit_norm = bool(args.get("eval_logit_norm", False))
        # self.eval_logit_norm_value = args.get("eval_logit_norm_value", args.get("logit_norm", 0.1))

    @property
    def feature_dim(self):
        return self.image_encoder.out_dim

    def extract_vector(self, image, task_id=None):
        if task_id is None:
            task_id = self.numtask - 1

        image_features = self.image_encoder(image, task_id)
        image_features = image_features[:, 0, :]
        return image_features

    def forward(self, image, get_feat=False, get_cur_feat=False, fc_only=False):
        if (not self.use_RP) or self.classifier_pool[self.numtask - 1].weight.requires_grad:
            if fc_only:
                fc_outs = []
                for ti in range(self.numtask):
                    fc_outs.append(
                        F.linear(
                            F.normalize(image, p=2, dim=1),
                            F.normalize(self.classifier_pool[ti].weight, p=2, dim=1),
                        )
                    )
                return torch.cat(fc_outs, dim=1)

            image_features = self.image_encoder(
                image,
                task_id=self.numtask - 1,
                get_feat=get_feat,
                get_cur_feat=get_cur_feat,
            )
            class_tokens = image_features[:, 0, :]
            class_tokens = class_tokens.view(class_tokens.size(0), -1)
            patch_tokens = image_features[:, 1:, :]

            logits = F.linear(
                F.normalize(class_tokens, p=2, dim=1),
                F.normalize(self.classifier_pool[self.numtask - 1].weight, p=2, dim=1),
            )

            return {
                "logits": logits,
                "features": class_tokens,
                "patch_tokens": patch_tokens,
            }

        image_features = self.image_encoder(
            image,
            task_id=self.numtask - 1,
            get_feat=get_feat,
            get_cur_feat=get_cur_feat,
        )
        class_tokens = image_features[:, 0, :]
        class_tokens = class_tokens.view(class_tokens.size(0), -1)
        patch_tokens = image_features[:, 1:, :]

        if self.W_rand is not None:
            inn = torch.nn.functional.relu(class_tokens @ self.W_rand)
        else:
            inn = class_tokens

        assert self.weight is not None
        logits = F.linear(inn, self.weight)
        return {
            "logits": logits,
            "features": class_tokens,
            "patch_tokens": patch_tokens,
        }

    def get_logits_per_task(self, image_features, with_grad=True):
        logits = []
        for head in self.classifier_pool[:self.numtask]:
            weight = F.normalize(head.weight, p=2, dim=1)
            if not with_grad:
                weight = weight.detach()
            logits.append(F.linear(F.normalize(image_features, p=2, dim=1), weight))
        return torch.stack(logits).permute(1, 0, 2)

    def interface(self, image, task_id=None):
        image_features = self.image_encoder(
            image,
            task_id=self.numtask - 1 if task_id is None else task_id, # 默认用当前已经训练到的最后一个 task id
        )
        image_features = image_features[:, 0, :] # [1,768]
        image_features = image_features.view(image_features.size(0), -1) # [1,768]

        logits = []
        for head in self.classifier_pool[:self.numtask]: # numtask 是当前已经训练到的 task id，classifier_pool[:self.numtask] 是所有已经训练过的分类头
            logits.append(
                F.linear(
                    F.normalize(image_features, p=2, dim=1),
                    F.normalize(head.weight, p=2, dim=1),
                )
            )

        # return torch.cat(logits, 1)
        logits = torch.cat(logits, dim=1)
        # if self.eval_logit_norm:
        #     task_norm = torch.norm(logits, p=2, dim=-1, keepdim=True) + 1e-7
        #     logits = logits / task_norm / float(self.eval_logit_norm_value)
        return logits

    def update_fc(self, nb_classes):
        self.numtask += 1

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self
