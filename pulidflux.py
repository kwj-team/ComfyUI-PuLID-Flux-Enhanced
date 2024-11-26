import torch
from torch import nn, Tensor
from torchvision import transforms
from torchvision.transforms import functional
import os
import logging
import folder_paths
import comfy.utils
from comfy.ldm.flux.layers import timestep_embedding
from insightface.app import FaceAnalysis
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

import torch.nn.functional as F

from .eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .encoders_flux import IDFormer, PerceiverAttentionCA

INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")

MODELS_DIR = os.path.join(folder_paths.models_dir, "pulid")
if "pulid" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["pulid"]
folder_paths.folder_names_and_paths["pulid"] = (current_paths, folder_paths.supported_pt_extensions)

from .online_train2 import online_train

class PulidFluxModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.double_interval = 2
        self.single_interval = 4

        # Init encoder
        self.pulid_encoder = IDFormer()

        # Init attention
        num_ca = 19 // self.double_interval + 38 // self.single_interval
        if 19 % self.double_interval != 0:
            num_ca += 1
        if 38 % self.single_interval != 0:
            num_ca += 1
        self.pulid_ca = nn.ModuleList([
            PerceiverAttentionCA() for _ in range(num_ca)
        ])

    def from_pretrained(self, path: str):
        state_dict = comfy.utils.load_torch_file(path, safe_load=True)
        state_dict_dict = {}
        for k, v in state_dict.items():
            module = k.split('.')[0]
            state_dict_dict.setdefault(module, {})
            new_k = k[len(module) + 1:]
            state_dict_dict[module][new_k] = v

        for module in state_dict_dict:
            getattr(self, module).load_state_dict(state_dict_dict[module], strict=True)

        del state_dict
        del state_dict_dict

    def get_embeds(self, face_embed, clip_embeds):
        return self.pulid_encoder(face_embed, clip_embeds)

def tensor_to_image(tensor):
    image = tensor.mul(255).clamp(0, 255).byte().cpu()
    image = image[..., [2, 1, 0]].numpy()
    return image

def image_to_tensor(image):
    tensor = torch.clamp(torch.from_numpy(image).float() / 255., 0, 1)
    tensor = tensor[..., [2, 1, 0]]
    return tensor

def resize_with_pad(img, target_size): # image: 1, h, w, 3
    img = img.permute(0, 3, 1, 2)
    H, W = target_size
    
    h, w = img.shape[2], img.shape[3]
    scale_h = H / h
    scale_w = W / w
    scale = min(scale_h, scale_w)

    new_h = int(min(h * scale,H))
    new_w = int(min(w * scale,W))
    new_size = (new_h, new_w)
    
    img = F.interpolate(img, size=new_size, mode='bicubic', align_corners=False)
    
    pad_top = (H - new_h) // 2
    pad_bottom = (H - new_h) - pad_top
    pad_left = (W - new_w) // 2
    pad_right = (W - new_w) - pad_left
    img = F.pad(img, pad=(pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    
    return img.permute(0, 2, 3, 1)

def to_gray(img):
    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    x = x.repeat(1, 3, 1, 1)
    return x

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

class PulidFluxModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"pulid_file": (folder_paths.get_filename_list("pulid"), )}}

    RETURN_TYPES = ("PULIDFLUX",)
    FUNCTION = "load_model"
    CATEGORY = "pulid"

    def load_model(self, pulid_file):
        model_path = folder_paths.get_full_path("pulid", pulid_file)

        # Also initialize the model, takes longer to load but then it doesn't have to be done every time you change parameters in the apply node
        model = PulidFluxModel()

        logging.info("Loading PuLID-Flux model.")
        model.from_pretrained(path=model_path)

        return (model,)

class PulidFluxInsightFaceLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["CPU", "CUDA", "ROCM"], ),
            },
        }

    RETURN_TYPES = ("FACEANALYSIS",)
    FUNCTION = "load_insightface"
    CATEGORY = "pulid"

    def load_insightface(self, provider):
        model = FaceAnalysis(name="antelopev2", root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider',]) # alternative to buffalo_l
        model.prepare(ctx_id=0, det_size=(640, 640))

        return (model,)

class PulidFluxEvaClipLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
        }

    RETURN_TYPES = ("EVA_CLIP",)
    FUNCTION = "load_eva_clip"
    CATEGORY = "pulid"

    def load_eva_clip(self):
        from .eva_clip.factory import create_model_and_transforms

        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)

        model = model.visual

        eva_transform_mean = getattr(model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            model["image_mean"] = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            model["image_std"] = (eva_transform_std,) * 3

        return (model,)
    
class UnApplyPulidFlux:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL", )}}
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "unapply_pulid_flux"
    CATEGORY = "pulid"
    
    def unapply_pulid_flux(self, model):
        flux_model = model.model.diffusion_model
        if hasattr(flux_model, "pulid_ca"):
            # Clear out the existing PuLID data with zero-weight entries
            flux_model.pulid_data = {key: {
                'weight': 0.0,  # Set weight to zero
                'embedding': torch.zeros_like(data['embedding']),  # Zero out embeddings
                'sigma_start': data['sigma_start'],  # Preserve original sigma values
                'sigma_end': data['sigma_end']
            } for key, data in flux_model.pulid_data.items()}
                    
            new_method = forward_orig.__get__(flux_model, flux_model.__class__)
            setattr(flux_model, 'forward_orig', new_method)
        return (model,)

class ApplyPulidFlux:
    @classmethod
    def INPUT_TYPES(s):  
        return {
            "required": {
                "model": ("MODEL", ),
                "pulid_flux": ("PULIDFLUX", ),
                "eva_clip": ("EVA_CLIP", ),
                "face_analysis": ("FACEANALYSIS", ),
                "image": ("IMAGE", ),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05 }),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "fusion": (["mean","concat","max","norm_id","max_token","auto_weight","train_weight"],),
                "fusion_weight_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1 }),
                "fusion_weight_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1 }),
                "train_step": ("INT", {"default": 1000, "min": 0, "max": 20000, "step": 1 }),
                "use_gray": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
            },
            "optional": {
                "attn_mask": ("MASK", ),
                "prior_image": ("IMAGE",), # for train weight, as the target
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_pulid_flux"
    CATEGORY = "pulid"

    def __init__(self):
        self.pulid_data_dict = None

    def apply_pulid_flux(self, model, pulid_flux, eva_clip, face_analysis, image, weight, start_at, end_at, prior_image=None,fusion="mean", fusion_weight_max=1.0, fusion_weight_min=0.0, train_step=1000, use_gray=True, attn_mask=None, unique_id=None):
        device = comfy.model_management.get_torch_device()
        dtype = model.model.diffusion_model.dtype
        if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            dtype = torch.bfloat16

        eva_clip.to(device, dtype=dtype)
        pulid_flux.to(device, dtype=dtype)

        if attn_mask is not None:
            if attn_mask.dim() > 3:
                attn_mask = attn_mask.squeeze(-1)
            elif attn_mask.dim() < 3:
                attn_mask = attn_mask.unsqueeze(0)
            attn_mask = attn_mask.to(device, dtype=dtype)

        if prior_image is not None:
            prior_image = resize_with_pad(prior_image.to(image.device, dtype=image.dtype), target_size=(image.shape[1], image.shape[2]))
            image=torch.cat((prior_image,image),dim=0)
        image = tensor_to_image(image)

        face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=device,
        )

        face_helper.face_parse = None
        face_helper.face_parse = init_parsing_model(model_name='bisenet', device=device)

        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        cond = []

        for i in range(image.shape[0]):
            iface_embeds = None
            for size in [(size, size) for size in range(640, 256, -64)]:
                face_analysis.det_model.input_size = size
                face_info = face_analysis.get(image[i])
                if face_info:
                    face_info = sorted(face_info, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[-1]
                    iface_embeds = torch.from_numpy(face_info.embedding).unsqueeze(0).to(device, dtype=dtype)
                    break
            else:
                logging.warning(f'Warning: No face detected in image {str(i)}')
                continue

            face_helper.clean_all()
            face_helper.read_image(image[i])
            face_helper.get_face_landmarks_5(only_center_face=True)
            face_helper.align_warp_face()

            if len(face_helper.cropped_faces) == 0:
                continue

            align_face = face_helper.cropped_faces[0]
            align_face = image_to_tensor(align_face).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            parsing_out = face_helper.face_parse(functional.normalize(align_face, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
            parsing_out = parsing_out.argmax(dim=1, keepdim=True)
            bg = sum(parsing_out == i for i in bg_label).bool()
            white_image = torch.ones_like(align_face)
            if use_gray:
                _align_face = to_gray(align_face)
            else:
                _align_face = align_face
            face_features_image = torch.where(bg, white_image, _align_face)

            face_features_image = functional.resize(face_features_image, eva_clip.image_size, transforms.InterpolationMode.BICUBIC if 'cuda' in device.type else transforms.InterpolationMode.NEAREST).to(device, dtype=dtype)
            face_features_image = functional.normalize(face_features_image, eva_clip.image_mean, eva_clip.image_std)

            id_cond_vit, id_vit_hidden = eva_clip(face_features_image, return_all_features=False, return_hidden=True, shuffle=False)
            id_cond_vit = id_cond_vit.to(device, dtype=dtype)
            for idx in range(len(id_vit_hidden)):
                id_vit_hidden[idx] = id_vit_hidden[idx].to(device, dtype=dtype)

            id_cond_vit = torch.div(id_cond_vit, torch.norm(id_cond_vit, 2, 1, True))

            id_cond = torch.cat([iface_embeds, id_cond_vit], dim=-1)

            cond.append(pulid_flux.get_embeds(id_cond, id_vit_hidden))

        if not cond:
            logging.warning("PuLID warning: No faces detected in any of the given images, returning unmodified model.")
            return (model,)

        if fusion == "mean":
            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                cond = torch.mean(cond, dim=0, keepdim=True)
        elif fusion == "concat":
            cond = torch.cat(cond, dim=1).to(device, dtype=dtype)
        elif fusion == "max":
            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                cond = torch.max(cond, dim=0, keepdim=True)[0]
        elif fusion == "norm_id":
            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                norm=torch.norm(cond,dim=(1,2))
                norm=norm/torch.sum(norm)
                cond=torch.einsum("wij,w->ij",cond,norm).unsqueeze(0)
        elif fusion == "max_token":
            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                norm=torch.norm(cond,dim=2)
                _,idx=torch.max(norm,dim=0)
                cond=torch.stack([cond[j,i] for i,j in enumerate(idx)]).unsqueeze(0)
        elif fusion == "auto_weight":
            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                norm=torch.norm(cond,dim=2)
                order=torch.argsort(norm,descending=False,dim=0)
                regular_weight=torch.linspace(fusion_weight_min,fusion_weight_max,norm.shape[0]).to(device, dtype=dtype)

                _cond=[]
                for i in range(cond.shape[1]):
                    o=order[:,i]
                    _cond.append(torch.einsum('ij,i->j',cond[:,i,:],regular_weight[o]))
                cond=torch.stack(_cond,dim=0).unsqueeze(0)
        elif fusion == "train_weight":
            cond = torch.cat(cond).to(device, dtype=dtype)
            if cond.shape[0] > 1:
                if train_step > 0:
                    with torch.inference_mode(False):
                        cond = online_train(cond, device=cond.device, step=train_step)
                else:
                    cond = torch.mean(cond, dim=0, keepdim=True)

        sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

        # Create a new ModelPatcher instance instead of cloning
        model_patcher = comfy.model_patcher.ModelPatcher(model.model, 
                                                        load_device=comfy.model_management.get_torch_device(),
                                                        offload_device=comfy.model_management.unet_offload_device())

        # Add PuLID components to the model if not already present
        if not hasattr(model.model.diffusion_model, "pulid_ca"):
            model_patcher.add_object_patch("pulid_ca", pulid_flux.pulid_ca)
            model_patcher.add_object_patch("pulid_double_interval", pulid_flux.double_interval)
            model_patcher.add_object_patch("pulid_single_interval", pulid_flux.single_interval)
            model_patcher.add_object_patch("pulid_data", {})

        # Store the current node's data
        pulid_data = model_patcher.get_model_object("pulid_data")
        pulid_data[unique_id] = {
            'weight': weight,
            'embedding': cond,
            'sigma_start': sigma_start,
            'sigma_end': sigma_end,
        }

        def pulid_wrapper(args):
            model_function = args["original_function"]
            flux_model = args["model"].diffusion_model
            
            # Extract needed values from args
            img = args["input"]
            txt = args.get("c_concat", [None])[0]
            timesteps = args["timesteps"]
            
            # Process double blocks
            ca_idx = 0
            if hasattr(flux_model, "pulid_ca"):
                # Process double blocks
                for i in range(len(flux_model.double_blocks)):
                    if i % flux_model.pulid_double_interval == 0:
                        for _, node_data in pulid_data.items():
                            condition_start = node_data['sigma_start'] >= timesteps
                            condition_end = timesteps >= node_data['sigma_end']
                            condition = torch.logical_and(condition_start, condition_end).all()
                            
                            if condition:
                                img = img + node_data['weight'] * flux_model.pulid_ca[ca_idx](node_data['embedding'], img)
                        ca_idx += 1

                # Process single blocks 
                for i in range(len(flux_model.single_blocks)):
                    if i % flux_model.pulid_single_interval == 0:
                        for _, node_data in pulid_data.items():
                            condition_start = node_data['sigma_start'] >= timesteps
                            condition_end = timesteps >= node_data['sigma_end']
                            condition = torch.logical_and(condition_start, condition_end).all()
                            
                            if condition:
                                img = img + node_data['weight'] * flux_model.pulid_ca[ca_idx](node_data['embedding'], img)
                        ca_idx += 1

            # Update input in args before calling original function
            args["input"] = img
            return model_function(args)

        # Set the wrapper function
        model_patcher.set_model_unet_function_wrapper(pulid_wrapper)

        # Keep track for cleanup
        self.pulid_data_dict = {'data': pulid_data, 'unique_id': unique_id}

        return (model_patcher,)

    def __del__(self):
        if self.pulid_data_dict:
            del self.pulid_data_dict['data'][self.pulid_data_dict['unique_id']]
            del self.pulid_data_dict


NODE_CLASS_MAPPINGS = {
    "PulidFluxModelLoader": PulidFluxModelLoader,
    "PulidFluxInsightFaceLoader": PulidFluxInsightFaceLoader,
    "PulidFluxEvaClipLoader": PulidFluxEvaClipLoader,
    "ApplyPulidFlux": ApplyPulidFlux,
    "UnApplyPulidFlux": UnApplyPulidFlux,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PulidFluxModelLoader": "Load PuLID Flux Model",
    "PulidFluxInsightFaceLoader": "Load InsightFace (PuLID Flux)",
    "PulidFluxEvaClipLoader": "Load Eva Clip (PuLID Flux)",
    "ApplyPulidFlux": "Apply PuLID Flux",
    "UnApplyPulidFlux": "UnApply PuLID Flux",
}
