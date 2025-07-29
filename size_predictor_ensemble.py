import torch
import numpy as np
import torch.nn as nn
from PIL import Image, ImageOps
import clip

class FinalPredictor(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=256, dropout_rate=0.4):
        super(FinalPredictor, self).__init__()
        self.global_img_proj = nn.Linear(embed_dim, hidden_dim)
        self.object_img_proj = nn.Linear(embed_dim, hidden_dim)
        self.text_exist_proj = nn.Linear(embed_dim, hidden_dim)
        self.text_wanted_proj = nn.Linear(embed_dim, hidden_dim)
        self.size_proj = nn.Linear(2, hidden_dim)
        self.interaction_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, global_img_feat, existing_object_feat, text_exist_feat, text_wanted_feat, exist_size):
        global_emb = self.global_img_proj(global_img_feat)
        obj_emb = self.object_img_proj(existing_object_feat)
        text_exist_emb = self.text_exist_proj(text_exist_feat)
        text_wanted_emb = self.text_wanted_proj(text_wanted_feat)
        size_emb = self.size_proj(exist_size)
        
        global_size_interaction = self.interaction_proj(torch.cat([global_emb, size_emb], dim=-1))
        object_size_interaction = self.interaction_proj(torch.cat([obj_emb, size_emb], dim=-1))
        text_exist_size_interaction = self.interaction_proj(torch.cat([text_exist_emb, size_emb], dim=-1))
        text_wanted_size_interaction = self.interaction_proj(torch.cat([text_wanted_emb, size_emb], dim=-1))
        
        combined = torch.cat([global_size_interaction, object_size_interaction, text_exist_size_interaction, text_wanted_size_interaction], dim=1)
        return self.fc_out(combined)

class SizePredictor:
    def __init__(self, model_paths, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        print("Loading CLIP model...")
        self.clip_model, self.preprocess_clip = clip.load("ViT-B/32", self.device)
        
        print("Loading size prediction models...")
        self.size_models = []
        for path in model_paths:
            print(f"  > Loading model from: {path}")
            # loading FinalPredictor
            model = FinalPredictor().to(self.device)
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            self.size_models.append(model)
        print("All models loaded successfully.")

    def _resize_image_with_padding(self, image, target_size=(512, 512)):
        original_width, original_height = image.size
        scale = min(target_size[0] / original_width, target_size[1] / original_height)
        new_width, new_height = int(original_width * scale), int(original_height * scale)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        padding_left = (target_size[0] - new_width) // 2
        padding_top = (target_size[1] - new_height) // 2
        padding_right = target_size[0] - new_width - padding_left
        padding_bottom = target_size[1] - new_height - padding_top
        padded_image = ImageOps.expand(resized_image, (padding_left, padding_top, padding_right, padding_bottom), fill=(0, 0, 0))
        return padded_image, scale, (padding_top, padding_bottom, padding_left, padding_right)

    def _pad_scale_bounding_box(self, bbox_xywh, scale, padding):
        x, y, w, h = bbox_xywh
        x_scaled = x * scale + padding[2]
        y_scaled = y * scale + padding[0]
        width_scaled = w * scale
        height_scaled = h * scale
        return [x_scaled, y_scaled, width_scaled, height_scaled]

    def _unpad_rescale_box(self, box_xyxy, original_size, padding, scale):
        padding_top, _, padding_left, _ = padding
        x1, y1, x2, y2 = box_xyxy
        x1_unpad = (x1 - padding_left) / scale
        y1_unpad = (y1 - padding_top) / scale
        x2_unpad = (x2 - padding_left) / scale
        y2_unpad = (y2 - padding_top) / scale
        H, W = original_size
        x1_final = max(min(x1_unpad, W), 0)
        y1_final = max(min(y1_unpad, H), 0)
        x2_final = max(min(x2_unpad, W), 0)
        y2_final = max(min(y2_unpad, H), 0)
        return [x1_final, y1_final, x2_final, y2_final]

    # Main prediction function
    def predict_size(self, image, existing_bbox, existing_object_text, target_object_text, direction):
        # 1. Convert bbox format from corners to [x, y, w, h]
        x1, y1, x2, y2 = existing_bbox #existing_bbox_corners
        existing_bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
        
        # 2. Preprocess full image and extract ROI features
        padded_img, scale, padding = self._resize_image_with_padding(image)
        if existing_bbox_xywh[2] <= 0 or existing_bbox_xywh[3] <= 0: return None
        existing_object_image = image.crop((x1, y1, x2, y2))
        
        # 3. Extract all features
        with torch.no_grad():
            processed_padded_image = self.preprocess_clip(padded_img).unsqueeze(0).to(self.device)
            processed_object_image = self.preprocess_clip(existing_object_image).unsqueeze(0).to(self.device)

            global_img_feat = self.clip_model.encode_image(processed_padded_image).to(torch.float32)
            existing_object_feat = self.clip_model.encode_image(processed_object_image).to(torch.float32)
            text_exist_feat = self.clip_model.encode_text(clip.tokenize([existing_object_text]).to(self.device)).to(torch.float32)
            text_wanted_feat = self.clip_model.encode_text(clip.tokenize([target_object_text]).to(self.device)).to(torch.float32)

        # 4. Compute normalized size
        scaled_bbox = self._pad_scale_bounding_box(existing_bbox_xywh, scale, padding)
        exist_w_norm = scaled_bbox[2] / 512.0
        exist_h_norm = scaled_bbox[3] / 512.0

        features = {
            "global_img_feat": global_img_feat, "existing_object_feat": existing_object_feat,
            "text_exist_feat": text_exist_feat, "text_wanted_feat": text_wanted_feat,
            "exist_size": torch.tensor([[exist_w_norm, exist_h_norm]], dtype=torch.float32).to(self.device)
        }

        # 5. Get predictions from all models and average them
        all_preds = []
        for model in self.size_models:
            with torch.no_grad():
                log_pred = model(**features)
                pred = torch.expm1(log_pred)
                all_preds.append(pred)
        
        avg_pred_ratio = torch.mean(torch.stack(all_preds), dim=0).squeeze().cpu().numpy()
        
        # 6. Compute final dimensions and bounding box
        pred_width = min(avg_pred_ratio[0] * scaled_bbox[2], 512)
        pred_height = min(avg_pred_ratio[1] * scaled_bbox[3], 512)
        pred_width, pred_height = max(pred_width, 50), max(pred_height, 50)

        scaled_x1, scaled_y1, scaled_w, scaled_h = scaled_bbox
        scaled_x2, scaled_y2 = scaled_x1 + scaled_w, scaled_y1 + scaled_h
        middle_x, half_pred_width = (scaled_x1 + scaled_x2) / 2, pred_width / 2

        if direction == "right":
            pred_x1, pred_y2, pred_y1 = int(scaled_x2), int(scaled_y2), int(scaled_y2 - pred_height)
            pred_x2 = int(min(pred_x1 + pred_width, 512))
        elif direction == "left":
            pred_x2, pred_y2, pred_y1 = int(scaled_x1), int(scaled_y2), int(scaled_y2 - pred_height)
            pred_x1 = int(max(pred_x2 - pred_width, 0))
        elif direction == "top":
            pred_x1 = int(max(middle_x - half_pred_width, 0))
            pred_y2 = int(scaled_y1)
            pred_x2 = int(min(middle_x + half_pred_width, 512))
            pred_y1 = int(max(pred_y2 - pred_height, 0))
        elif direction == "bottom":
            pred_x1 = int(max(middle_x - half_pred_width, 0))
            pred_y1 = int(scaled_y2)
            pred_x2 = int(min(middle_x + half_pred_width, 512))
            pred_y2 = int(min(pred_y1 + pred_height, 512))
        else:
            # Default centered position if direction not recognized
            pred_x1 = int(max(middle_x - half_pred_width, 0))
            pred_y1 = int(max(scaled_y2 + 10, 0))  # 10px below existing box
            pred_x2 = int(min(pred_x1 + pred_width, 512))
            pred_y2 = int(min(pred_y1 + pred_height, 512))
        
        # Convert back to original image space
        pred_box_512 = [pred_x1, pred_y1, pred_x2, pred_y2]
        pred_box_real = self._unpad_rescale_box(pred_box_512, image.size[::-1], padding, scale)
        
        return {
            "predicted_size_ratio": avg_pred_ratio,
            "predicted_box_padded": pred_box_512,
            "predicted_box_original": pred_box_real,
            "padding": padding,
            "scale": scale
        }