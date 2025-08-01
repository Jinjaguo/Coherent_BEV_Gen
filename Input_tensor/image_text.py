import types, sys, torch
if not hasattr(torch, "xpu"):          # 没有 xpu 就伪造一个
    dummy = types.ModuleType("torch.xpu")
    class _DummyXPU:
        def __getattr__(self, name):
            raise AttributeError("torch.xpu is not available in this build!")
    dummy.device = _DummyXPU()
    torch.xpu = dummy                   # 挂到 torch
    sys.modules["torch.xpu"] = dummy    # 供 import torch.xpu

import os
import open_clip
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np

image_folder = r"C:\Users\Guoji\Desktop\Files\data\mini\samples\CAM_FRONT"
save_dir     = "../text_embeddings";  os.makedirs(save_dir, exist_ok=True)
model_path   = r"../cogvlm2"
torch_type = torch.float16

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==== 加载模型 ====
caption_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_type,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).eval()
caption_model = caption_model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# ==== 加载 CLIP ====
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai", device=device)
clip_model.eval()

history = []
results = {}
for fn in os.listdir(image_folder):
    if not fn.lower().endswith(".jpg"):
        continue
    img_path = os.path.join(image_folder, fn)
    image = Image.open(img_path).convert("RGB")

    # ---------- Prompt ----------
    prompt = (
        "Follow the examples below to describe the driving scenario in one sentence.\n"
        "Example 1: An urban road passing through a block of red brick buildings, with a red light ahead, few pedestrians and vehicles, and cloudy weather.\n"
        "Example 2: A city street flanked by red brick buildings, with cars approaching the intersection and pedestrians crossing under cloudy weather.\n"
        "Example 3: An urban intersection with red brick buildings, traffic lights showing red, and a few pedestrians on sidewalks under cloudy skies.\n"
        "Example 4: A red brick urban block with cars moving forward at a yellow light, scattered pedestrians, and an overcast sky."
    )

    inputs = caption_model.build_conversation_input_ids(tokenizer, query=prompt, history=[], images=[image])  # chat mode
    # ---------- AutoProcessor 统一打包 ----------
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(torch_type)]],
    }

    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
        "top_k": 1,
    }

    with torch.no_grad():
        outputs = caption_model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("📷", fn, "→", caption)

    # ---------- CLIP embedding ----------
    txt_tok = open_clip.tokenize([caption]).to(device)
    with torch.no_grad():
        txt_vec = clip_model.encode_text(txt_tok)
        txt_vec /= txt_vec.norm(dim=-1, keepdim=True)

    # ---------- 保存 ----------
    base = os.path.splitext(fn)[0]
    npy_path = os.path.join(save_dir, f"{base}.npy")
    np.save(npy_path, txt_vec.cpu().numpy())
    results[base] = {"caption": caption, "embedding_path": npy_path}

# 总索引
with open(os.path.join(save_dir, "index.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n🎉 All images processed & embeddings saved! 🚀")