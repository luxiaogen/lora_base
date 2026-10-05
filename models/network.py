import torch
from torch import nn
from torch.nn import functional as F
try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_

from models.vit import VisionTransformer, PatchEmbed, resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn
from models.attention import Attention_LoRA


class ViT(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, attn_fn=Attention_LoRA, n_tasks=10, rank=64):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, attn_fn=attn_fn, n_tasks=n_tasks, rank=rank)

    def forward(self, x, task_id):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        for i,blk in enumerate(self.blocks):
            x = blk(x, task_id)

        x = self.norm(x)

        return x



def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    pretrained_cfg = resolve_pretrained_cfg(variant)
    default_num_classes = pretrained_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    model = build_model_with_cfg(
        ViT, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model



class MANet(nn.Module):
    def __init__(self, args):
        super(MANet, self).__init__()

        model_kwargs = dict(
            patch_size=16,
            embed_dim=args["embd_dim"],
            depth=12,
            num_heads=args.get("num_heads", 12),
            n_tasks=args["total_sessions"],
            rank=args["rank"],
            attn_fn=Attention_LoRA,
        )
        self.image_encoder =_create_vision_transformer('vit_base_patch16_224_in21k', pretrained=True, **model_kwargs)

        self.class_num = args["init_cls"]
        self.dim = args["embd_dim"]
        self.classifier_pool = nn.ModuleList([
            nn.Linear(args["embd_dim"], self.class_num, bias=False)
            for i in range(args["total_sessions"])
        ])
        for m in self.classifier_pool.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)

        self.numtask = 0

    @property
    def feature_dim(self):
        return self.image_encoder.out_dim


    def extract_vector(self, image,  task_id=None):
        if task_id is None:
            task_id = self.numtask-1

        image_features = self.image_encoder(image, task_id)
        image_features = image_features[:,0,:]
        return image_features


    def forward(self, image, fc_only=False):
        if fc_only:
            logits = []
            for task in range(self.numtask):
                logits.append(F.linear(
                    F.normalize(image, p=2, dim=1),
                    F.normalize(self.classifier_pool[task].weight, p=2, dim=1)))
            return torch.cat(logits, dim=1)
        features = self.image_encoder(image, task_id=self.numtask - 1)
        class_tokens = features[:, 0, :].view(features.size(0), -1)
        logits = F.linear(F.normalize(class_tokens, p=2, dim=1),
                          F.normalize(self.classifier_pool[self.numtask - 1].weight, p=2, dim=1))
        return {"logits": logits, "features": class_tokens, "patch_tokens": features[:, 1:, :]}

    def interface(self, image):
        features = self.image_encoder(image, task_id=self.numtask - 1)[:, 0, :]
        features = features.view(features.size(0), -1)
        logits = []
        for head in self.classifier_pool[:self.numtask]:
            logits.append(F.linear(F.normalize(features, p=2, dim=1),
                                    F.normalize(head.weight, p=2, dim=1)))
        return torch.cat(logits, dim=1)

    def update_fc(self, nb_classes):
        self.numtask += 1
