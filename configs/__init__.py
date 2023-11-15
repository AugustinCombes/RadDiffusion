import transformers
import json
import os
current_dir = os.path.dirname(os.path.abspath(__file__))


xs_config_path = os.path.join(current_dir, 'vit_XS.json')
with open(xs_config_path, "r") as file:
    xs_config_dict = json.load(file)
xs_config = transformers.ViTMAEConfig.from_dict(xs_config_dict)


s_config_path = os.path.join(current_dir, 'vit_S.json')
with open(s_config_path, "r") as file:
    s_config_dict = json.load(file)
s_config = transformers.ViTMAEConfig.from_dict(s_config_dict)


b_config_path = os.path.join(current_dir, 'vit_B.json')
with open(b_config_path, "r") as file:
    b_config_dict = json.load(file)
b_config = transformers.ViTMAEConfig.from_dict(b_config_dict)


configs = {
    "xs": xs_config,
    "s": s_config,
    "b": b_config,
}