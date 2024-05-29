import os
from .clip_encoder import CLIPVisionTower
from .languagebind import LanguageBindImageTower, LanguageBindVideoTower
from .mae_encoder import MAEVisionTower
from transformers import CLIPModel

def build_image_tower(image_tower_cfg, **kwargs):
    image_tower = getattr(image_tower_cfg, 'mm_image_tower', getattr(image_tower_cfg, 'image_tower', None))
    # is_absolute_path_exists = os.path.exists(image_tower)
    is_absolute_path_exists = False
    if is_absolute_path_exists or image_tower.startswith("openai") or image_tower.startswith("laion"):
        return CLIPVisionTower(image_tower, args=image_tower_cfg, **kwargs)
    if image_tower.endswith('LanguageBind_Image'):
        return LanguageBindImageTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)
    if 'mae' in image_tower:
        print('maemaemaemaemaemaemaemae')
        print('maemaemaemaemaemaemaemae')
        print('maemaemaemaemaemaemaemae')
        print('maemaemaemaemaemaemaemae')
        print('maemaemaemaemaemaemaemae')
        return MAEVisionTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)
    raise ValueError(f'Unknown image tower: {image_tower}')

def build_video_tower(video_tower_cfg, **kwargs):
    video_tower = getattr(video_tower_cfg, 'mm_video_tower', getattr(video_tower_cfg, 'video_tower', None))

    if video_tower.endswith('LanguageBind_Video_merge'):
        return LanguageBindVideoTower(video_tower, args=video_tower_cfg, cache_dir='./cache_dir', **kwargs)
    raise ValueError(f'Unknown video tower: {video_tower}')

def extractor(**kwargs):
    video_tower = "./cache_dir/LanguageBind_Video_merge"
    class VideoTowerConfig:
        def __init__(self):
            self.mm_video_tower = "./cache_dir/LanguageBind_Video_merge"
            self.mm_vision_select_feature = "patch"
            self.mm_vision_select_layer = -2
            self.model_type = "llava"
            self.num_attention_heads = 32
            self.num_hidden_layers = 32
            self.num_key_value_heads = 32
            self.pad_token_id = 0
            self.pretraining_tp = 1
            self.rms_norm_eps = 1e-05
            self.vocab_size = 32000

    video_tower_cfg = VideoTowerConfig()
    return LanguageBindVideoTower(video_tower, args=video_tower_cfg, cache_dir='./cache_dir', **kwargs)


# import os
# from .clip_encoder import CLIPVisionTower
# from .languagebind import LanguageBindImageTower, LanguageBindVideoTower
# from transformers import CLIPModel

# def build_image_tower(image_tower_cfg, **kwargs):
#     image_tower = getattr(image_tower_cfg, 'mm_image_tower', getattr(image_tower_cfg, 'image_tower', None))
#     is_absolute_path_exists = os.path.exists(image_tower)
#     if is_absolute_path_exists or image_tower.startswith("openai") or image_tower.startswith("laion"):
#         return CLIPVisionTower(image_tower, args=image_tower_cfg, **kwargs)
#     if image_tower.endswith('LanguageBind_Image'):
#         return LanguageBindImageTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)
#     raise ValueError(f'Unknown image tower: {image_tower}')

# def build_video_tower(video_tower_cfg, **kwargs):
#     video_tower = getattr(video_tower_cfg, 'mm_video_tower', getattr(video_tower_cfg, 'video_tower', None))
#     if video_tower.endswith('LanguageBind_Video'):
#         return LanguageBindVideoTower(video_tower, args=video_tower_cfg, cache_dir='./cache_dir', **kwargs)
#     raise ValueError(f'Unknown video tower: {video_tower}')