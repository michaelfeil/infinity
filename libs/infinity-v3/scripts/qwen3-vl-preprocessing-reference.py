"""Generate CPU reference fixtures for the ignored Rust native-processor test.

Requires transformers==5.17.0, torch==2.11.0, torchvision==0.26.0,
Pillow==12.3.0, numpy, and a local Qwen/Qwen3-VL-Embedding-2B checkpoint
at revision 9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda.

python scripts/qwen3-vl-preprocessing-reference.py CHECKPOINT OUTPUT
QWEN3_VL_CHECKPOINT=CHECKPOINT QWEN3_VL_FIXTURES=OUTPUT \
  cargo test -p infinity_v3-core --lib --features infinity_v3-backend/libtorch \
  native_processor_matches_reference -- --ignored
"""
import hashlib, json, types
from pathlib import Path
import numpy as np
import torch, torchvision, transformers
from PIL import Image, __version__ as pillow_version
from transformers import AutoProcessor, AutoConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from torchvision.transforms.v2 import functional as F

import argparse
parser = argparse.ArgumentParser(description="Generate pinned Qwen3-VL CPU processor fixtures")
parser.add_argument("checkpoint", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
ROOT, CHECKPOINT = args.output, args.checkpoint
ROOT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
processor = AutoProcessor.from_pretrained(CHECKPOINT, local_files_only=True)
config = AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True)
stub = types.SimpleNamespace(config=config)
stub.get_vision_position_ids = types.MethodType(Qwen3VLModel.get_vision_position_ids, stub)

def pattern(w, h, mode):
    y,x = np.indices((h,w),dtype=np.uint32)
    rgb = np.stack(((x*17+y*3)%256,(x*5+y*29)%256,((x^y)*23)%256),-1).astype('uint8')
    image = Image.fromarray(rgb)
    if mode == 'RGBA':
        image = image.convert('RGBA')
        image.putalpha(Image.fromarray(((x+y)*7%256).astype('uint8')))
    elif mode != 'RGB':
        image = image.convert(mode)
    return image

specs = [('unchanged',64,96,'RGB'),('upscale',17,23,'RGB'),('odd_landscape',173,95,'RGB'),('odd_portrait',95,173,'RGB'),('downscale',1503,1101,'RGB'),('grayscale',81,117,'L'),('rgba',117,81,'RGBA'),('palette',83,115,'P'),('round_tie',80,80,'RGB')]
manifest = dict(model='Qwen/Qwen3-VL-Embedding-2B',revision='9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda',torch=torch.__version__,torchvision=torchvision.__version__,transformers=transformers.__version__,pillow=pillow_version,processor_class=type(processor.image_processor).__name__,cases={})
images = {}
for name,w,h,mode in specs:
    image = pattern(w,h,mode)
    image.save(ROOT/f'{name}.png')
    images[name]=image
    rh,rw=smart_resize(h,w,factor=32,min_pixels=4096,max_pixels=1310720)
    rgb=image.convert('RGB')
    tensor=F.pil_to_tensor(rgb)
    reference_resize=F.resize(tensor,[rh,rw],interpolation=F.InterpolationMode.BICUBIC,antialias=True)
    expected=processor.image_processor(images=[image],return_tensors='pt')
    pixels=expected['pixel_values'].float().numpy()
    pixels.astype('<f4').tofile(ROOT/f'{name}.pixels.f32')
    np.asarray(rgb).tofile(ROOT/f'{name}.decoded.rgb8')
    reference_resize.permute(1,2,0).numpy().tofile(ROOT/f'{name}.resized.rgb8')
    pillow=np.asarray(rgb.resize((rw,rh),Image.Resampling.BICUBIC)).astype('int16')
    diff=pillow-reference_resize.permute(1,2,0).numpy().astype('int16')
    manifest['cases'][name]=dict(source_dimensions=[w,h],mode=mode,resized_dimensions=[rw,rh],grid=expected['image_grid_thw'].tolist(),pixels_shape=list(pixels.shape),visual_tokens=int(expected['image_grid_thw'].prod()//4),pillow_vs_torchvision_max_abs=int(abs(diff).max()),pillow_vs_torchvision_different_channels=int(np.count_nonzero(diff)),pixels_sha256=hashlib.sha256(pixels.astype('<f4').tobytes()).hexdigest())

for name, selected in [('one_image',['odd_landscape']),('two_images',['odd_portrait','rgba'])]:
    content=[{'type':'text','text':'Compare these images: '}]
    for i,key in enumerate(selected):
        content.extend([{'type':'image','image':images[key]},{'type':'text','text':f' Image {i+1}. '}])
    messages=[{'role':'user','content':content},{'role':'assistant','content':'I see the images.'},{'role':'user','content':'Represent their content.'}]
    inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
    positions,deltas=Qwen3VLModel.get_rope_index(stub,input_ids=inputs['input_ids'],mm_token_type_ids=inputs['mm_token_type_ids'],image_grid_thw=inputs['image_grid_thw'],attention_mask=inputs['attention_mask'])
    output={k:v.tolist() for k,v in inputs.items() if torch.is_tensor(v) and k!='pixel_values'}
    output.update(position_ids=positions.tolist(),mrope_position_deltas=deltas.tolist(),image_names=selected,rendered_prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
    (ROOT/f'{name}.sequence.json').write_text(json.dumps(output,indent=2)+'\n')
    manifest['cases'][name]=dict(input_tokens=inputs['input_ids'].shape[-1],grid=inputs['image_grid_thw'].tolist(),position_deltas=deltas.tolist())

# A fast implementation's fused arithmetic differs slightly from the draft's
# x * rescale_factor, subtract mean, divide std ordering even before resizing.
values=torch.arange(256,dtype=torch.float32)
draft=(values*float(processor.image_processor.rescale_factor)-0.5)/0.5
reference=(values-127.5)/127.5
manifest['normalization']=dict(draft_max_abs=float((draft-reference).abs().max()),different_values=int((draft!=reference).sum()),reference_expression='(float32(pixel)-127.5)/127.5')
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
