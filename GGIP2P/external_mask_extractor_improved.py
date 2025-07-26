import cv2
import torch
import spacy
import numpy as np
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download
from segment_anything import build_sam, SamPredictor
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import load_image, predict
import re
import inflect
import clip

# Import the separated components
# from size_predictor import SizePredictor
from size_predictor_ensemble import SizePredictor
from target_predictor import TargetPredictor

p = inflect.engine()
nlp = spacy.load("en_core_web_sm")

# Define spatial words for directional logic
spatial_words = {
    'left': ['left'],
    'right': ['right'],
    'top': ['top', 'upper', 'above', 'up', 'over'],
    'bottom': ['bottom', 'lower', 'below', 'down', 'under', 'underneath', 'beneath']
}

# Define pronouns and their replacements
male_pronouns = ['he', 'him', 'his']
female_pronouns = ['she', 'her']
neutral_pronouns = ['they', 'them', 'their']
neutral_replacements = ['people', 'animals', 'them']

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cuda:0'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    return model.eval().to(device)

def join_bboxes(bboxes_xyxy):
    min_x, min_y, _, _ = torch.min(bboxes_xyxy, dim=0)[0]
    _, _, max_x, max_y = torch.max(bboxes_xyxy, dim=0)[0]
    return torch.tensor([min_x.item(), min_y.item(), max_x.item(), max_y.item()])

def is_pronoun(word):
    doc = nlp(word)
    return doc[0].pos_ == "PRON"

def compute_intersection_ratio(box1, box2):
    """
    Computes intersection-to-box2-area ratio.
    Args:
        box1: List or tensor [x1, y1, x2, y2] for the larger box.
        box2: List or tensor [x1, y1, x2, y2] for the smaller box.
    Returns:
        containment_ratio: Ratio of intersection area to box2's area.
    """
    # Compute intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute intersection area
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    # Compute areas of each box
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # Compute intersection-to-box2-area ratio
    containment_ratio = inter_area / float(box2_area) if box2_area > 0 else 0
    return containment_ratio

def filter_boxes(boxes_xyxy, is_singular_target):
    """Filter boxes based on containment and singular/plural logic."""
    if boxes_xyxy.shape[0] <= 1:
        print("Only one box found!")
        return boxes_xyxy
    keep = [True] * boxes_xyxy.shape[0]
    for i in range(boxes_xyxy.shape[0]):
        for j in range(boxes_xyxy.shape[0]):
            if i != j:          
                containment_ratio = compute_intersection_ratio(boxes_xyxy[i], boxes_xyxy[j])
                if containment_ratio > 0.9:
                    area_i = (boxes_xyxy[i, 2] - boxes_xyxy[i, 0]) * (boxes_xyxy[i, 3] - boxes_xyxy[i, 1])
                    area_j = (boxes_xyxy[j, 2] - boxes_xyxy[j, 0]) * (boxes_xyxy[j, 3] - boxes_xyxy[j, 1])
                    if is_singular_target:
                        keep[i if area_i > area_j else j] = False # Remove larger box
                    else:                        
                        keep[i if area_i < area_j else j] = False # Remove smaller box
    return boxes_xyxy[torch.tensor(keep)]

def apply_directional_logic(boxes_xyxy, direction):
    if boxes_xyxy.shape[0] > 1:
        print("apply_directional_logic", direction)
        if direction == 'left':
            idx = torch.argmin(boxes_xyxy[:, 0])
            return boxes_xyxy[idx].unsqueeze(0)
        elif direction == 'right':
            idx = torch.argmax(boxes_xyxy[:, 2])
            return boxes_xyxy[idx].unsqueeze(0)
        elif direction == 'top':
            idx = torch.argmin(boxes_xyxy[:, 1])
            return boxes_xyxy[idx].unsqueeze(0)
        elif direction == 'bottom':
            idx = torch.argmax(boxes_xyxy[:, 3])
            return boxes_xyxy[idx].unsqueeze(0)
    return boxes_xyxy

def apply_relative_directional_logic(boxes_xyxy1, boxes_xyxy2, direction):
    print("Apply_relative_directional_logic", direction)
    if boxes_xyxy2.shape[0] > 1:
        if direction == 'left':
            idx = torch.argmin(torch.abs(boxes_xyxy2[:, 2] - boxes_xyxy1[:, 0]) + torch.abs((boxes_xyxy2[:, 1] + boxes_xyxy2[:, 3]) / 2 - (boxes_xyxy1[:, 1] + boxes_xyxy1[:, 3]) / 2))
        elif direction == 'right':
            idx = torch.argmin(torch.abs(boxes_xyxy2[:, 0] - boxes_xyxy1[:, 2]) + torch.abs((boxes_xyxy2[:, 1] + boxes_xyxy2[:, 3]) / 2 - (boxes_xyxy1[:, 1] + boxes_xyxy1[:, 3]) / 2))
        elif direction == 'top':
            idx = torch.argmin(torch.abs(boxes_xyxy2[:, 3] - boxes_xyxy1[:, 1]) + torch.abs((boxes_xyxy2[:, 0] + boxes_xyxy2[:, 2]) / 2 - (boxes_xyxy1[:, 0] + boxes_xyxy1[:, 2]) / 2))
        elif direction == 'bottom':
            idx = torch.argmin(torch.abs(boxes_xyxy2[:, 1] - boxes_xyxy1[:, 3]) + torch.abs((boxes_xyxy2[:, 0] + boxes_xyxy2[:, 2]) / 2 - (boxes_xyxy1[:, 0] + boxes_xyxy1[:, 2]) / 2))
        else:
            return boxes_xyxy2  # Return all if no valid direction is found
        return boxes_xyxy2[idx].unsqueeze(0)
    return boxes_xyxy2

class ExternalMaskExtractor:
    def __init__(self, 
                 device, 
                 target_predictor_model_path="/content/drive/MyDrive/BERT_TUNE/best_target_finder_big_lora_model",
                 size_predictor_model_path=["/content/drive/MyDrive/clean_imgcrop_model.pth",
                                            "/content/drive/MyDrive/clean_imgcrop_combloss_model.pth"],
                 sam_path='/content/drive/MyDrive/SAM/sam_vit_h_4b8939.pth'):
        self.device = device
        
        # Initialize the target predictor
        self.target_predictor = TargetPredictor(target_predictor_model_path)
        
        # Initialize the size predictor
        self.size_predictor = SizePredictor(size_predictor_model_path, device)
        
        # Initialize GroundingDINO
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        self.groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename)
        
        # Initialize SAM
        sam = build_sam(checkpoint=sam_path).to(device)
        self.sam_predictor = SamPredictor(sam)
        
        # Initialize CLIP for pronoun replacement
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)

    def _clip_metric(self, image, text):
        """Calculate CLIP similarity score between image and text"""
        preproc_image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        text_token = clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(preproc_image)
            text_features = self.clip_model.encode_text(text_token)
            
            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = (image_features @ text_features.T)[0]
            
        return similarity

    def replace_pronoun(self, target, target_phrase, image, verbose=False):
        """Replace pronouns with appropriate nouns based on GroundingDINO detection scores"""
        target_lower = target.lower()
        
        # Check if target is a pronoun
        if target_lower in male_pronouns:
            new_target = "man"
            new_target_phrase = target_phrase.replace(target, new_target)
            if verbose:
                print(f"Replaced '{target}' with '{new_target}'")
            return new_target, new_target_phrase
            
        elif target_lower in female_pronouns:
            new_target = "woman"
            new_target_phrase = target_phrase.replace(target, new_target)
            if verbose:
                print(f"Replaced '{target}' with '{new_target}'")
            return new_target, new_target_phrase
            
        elif target_lower in neutral_pronouns:
            # Calculate GroundingDINO detection scores for each possible replacement
            dino_scores = {}
            
            for noun in neutral_replacements:
                # Try to detect the noun using GroundingDINO
                boxes, scores, phrases = self._dino_predict(image, noun)
                
                if boxes is not None and boxes.shape[0] > 0:
                    # Convert boxes to xyxy format
                    H, W = image.size[::-1]  # PIL image has (width, height), we need (height, width)
                    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
                    
                    # Track which indices are kept after filtering
                    keep = [True] * boxes_xyxy.shape[0]
                    
                    # Apply the same filtering logic as in filter_boxes without calling the function
                    for i in range(boxes_xyxy.shape[0]):
                        for j in range(boxes_xyxy.shape[0]):
                            if i != j:
                                containment_ratio = compute_intersection_ratio(boxes_xyxy[i], boxes_xyxy[j])
                                if containment_ratio > 0.9:
                                    area_i = (boxes_xyxy[i, 2] - boxes_xyxy[i, 0]) * (boxes_xyxy[i, 3] - boxes_xyxy[i, 1])
                                    area_j = (boxes_xyxy[j, 2] - boxes_xyxy[j, 0]) * (boxes_xyxy[j, 3] - boxes_xyxy[j, 1])
                                    # False indicates not singular (plural), so remove smaller box
                                    keep[i if area_i > area_j else j] = False
                    
                    # Calculate total score only for boxes that are kept
                    total_score = sum(scores[i].item() for i in range(len(keep)) if keep[i])
                    num_boxes = sum(keep)
                    
                    dino_scores[noun] = total_score
                    
                    if verbose:
                        print(f'GroundingDINO Score("{noun}"): {total_score} from {num_boxes} boxes')
                else:
                    dino_scores[noun] = 0.0
                    if verbose:
                        print(f'GroundingDINO Score("{noun}"): 0.0 (no detections)')
            
            # Choose replacement with highest detection score
            if any(dino_scores.values()):  # If at least one replacement had detections
                best_noun = max(dino_scores, key=dino_scores.get)
                new_target_phrase = target_phrase.replace(target, best_noun)
                if verbose:
                    print(f"Replaced '{target}' with '{best_noun}' (highest GroundingDINO score: {dino_scores[best_noun]})")
                return best_noun, new_target_phrase
            else:
                # Fallback to "people" if no good detections (default choice)
                default_noun = "them"
                new_target_phrase = target_phrase.replace(target, default_noun)
                if verbose:
                    print(f"No detections found with GroundingDINO, defaulting to '{default_noun}'")
                return default_noun, new_target_phrase
                
        return target, target_phrase

    def _dino_predict(self, image, prompt):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_transformed, _ = transform(image, None)
        boxes, scores, phrases = predict(model=self.groundingdino_model, image=image_transformed, caption=prompt,
                              box_threshold=0.3, text_threshold=0.25, device='cuda:0')
        return (boxes, scores, phrases) if boxes.shape[0] > 0 else (None, None, None)

    def _grounded_sam_predict(self, image, first_target_phrase, first_target, second_target_phrase, second_target, final_sentence, direction):                    
        direction_is_used = False  # Initialize to avoid UnboundLocalError        
        image_orig = np.asarray(image)
        H, W, _ = image_orig.shape

        # Default black mask initialization
        black_mask = torch.zeros((H, W), dtype=torch.float32).to(self.device)
        predicted_size_mask = black_mask  # Initialize predicted_size_mask with black_mask

        # Matching spatial_words direction to 4 major directions if exists
        direction = next((key for key, vals in spatial_words.items() if any(val in direction.lower() for val in vals)), None)
        if direction:
            print(f"Matched direction: {direction}")
            direction_is_used = False
            print("Direction_is_used: ", direction_is_used)

        # *** MODIFICATION START ***
        # Call _dino_predict with both original "the" prefixed and not the target phrases for first target
        all_boxes1 = []
        all_scores1 = []
        all_phrases1 = []

        # Predict with current first_target_phrase
        boxes1_curr, scores1_curr, phrases1_curr = self._dino_predict(image, first_target_phrase)
        if boxes1_curr is not None:
            print("Shape of Boxes for", first_target_phrase, " : ", boxes1_curr.shape)
            all_boxes1.append(boxes1_curr)
            all_scores1.append(scores1_curr)
            all_phrases1.append(phrases1_curr)

        # Predict with "the" prefixed first_target_phrase if not already prefixed
        # if not first_target_phrase.lower().startswith("the "):
            # the_first_target_phrase = "the " + first_target_phrase
        if first_target_phrase.lower().startswith("the "):
            notthe_first_target_phrase = first_target_phrase[4:].strip()
            boxes1_notthe, scores1_notthe, phrases1_notthe = self._dino_predict(image, notthe_first_target_phrase)
            if boxes1_notthe is not None:
                print("Shape of Boxes for", notthe_first_target_phrase, " (without the) : ", boxes1_notthe.shape)
                all_boxes1.append(boxes1_notthe)
                all_scores1.append(scores1_notthe)
                all_phrases1.append(phrases1_notthe)
        
        if not all_boxes1:
            return torch.zeros(image_orig.shape[:2], dtype=torch.float32).to('cuda:0'), None, black_mask # Return black mask if no detections
        
        # Concatenate all boxes and convert to xyxy format
        boxes1 = torch.cat(all_boxes1)
        scores1 = torch.cat(all_scores1)
        phrases1 = [p for sublist in all_phrases1 for p in sublist] # Flatten the list of phrases

        boxes_xyxy1 = box_ops.box_cxcywh_to_xyxy(boxes1) * torch.Tensor([W, H, W, H])
        # *** MODIFICATION END ***

        # # Clean bounding box coordinates
        # boxes_xyxy1 = torch.where(boxes_xyxy1 < 1, torch.tensor(0.0).to(self.device), boxes_xyxy1)
        # boxes_xyxy1[:, [0, 2]] = torch.clamp(boxes_xyxy1[:, [0, 2]], 0, W)  # Clamp x1, x2 to [0, W]
        # boxes_xyxy1[:, [1, 3]] = torch.clamp(boxes_xyxy1[:, [1, 3]], 0, H)  # Clamp y1, y2 to [0, H]
        # print(f"Cleaned boxes_xyxy1 for '{first_target_phrase}': {boxes_xyxy1.tolist()}")
##################################################################################################

        # Check if the target object is singular or plural
        first_target_is_singular = p.singular_noun(first_target.lower()) is False
        print("ALL Boxes xyxy1 Shape: ", boxes_xyxy1.shape)
        print(first_target, " is_singular? --> ", first_target_is_singular)

        # Filter boxes by Intersection if multiple are detected
        if boxes_xyxy1.shape[0] > 1:
            boxes_xyxy1 = filter_boxes(boxes_xyxy1, first_target_is_singular)
            print("Filtered Boxes1 Shape: ", boxes_xyxy1.shape)

        # Apply directional logic if multiple boxes remain and direction is provided
        if boxes_xyxy1.shape[0] > 1 and direction:
            boxes_xyxy1 = apply_directional_logic(boxes_xyxy1, direction)
            direction_is_used = True
            print("Directed Filtered Boxes1 Shape: ", boxes_xyxy1.shape, "Direction_is_used? ", direction_is_used)
            
        # Check if any directional word from matched direction appears in first_target_phrase
        if direction and any(word in first_target_phrase.lower() for word in spatial_words.get(direction, [])):
            direction_is_used = True
            print("Direction word found in first_target_phrase. direction_is_used = True")

        boxes_xyxy = boxes_xyxy1
        final_target = first_target_phrase
        print("boxes_xyxy1: ", boxes_xyxy1)

        if second_target != "No second target found":
            # *** MODIFICATION START ***
            # Call _dino_predict with both original and "the" prefixed target phrases for second target
            all_boxes2 = []
            all_scores2 = []
            all_phrases2 = []

            # Predict with current second_target_phrase
            boxes2_curr, scores2_curr, phrases2_curr = self._dino_predict(image, second_target_phrase)
            if boxes2_curr is not None:
                print("Shape of Boxes for", second_target_phrase, " (without the) : ", boxes2_curr.shape)
                all_boxes2.append(boxes2_curr)
                all_scores2.append(scores2_curr)
                all_phrases2.append(phrases2_curr)

            # Predict with "the" prefixed second_target_phrase if not already prefixed
            # if not second_target_phrase.lower().startswith("the "):
            #     the_second_target_phrase = "the " + second_target_phrase
            if second_target_phrase.lower().startswith("the "):
                notthe_second_target_phrase = second_target_phrase[4:].strip()
                boxes2_notthe, scores2_notthe, phrases2_notthe = self._dino_predict(image, notthe_second_target_phrase)
                if boxes2_notthe is not None:
                    print("Shape of Boxes for", notthe_second_target_phrase, " : ", boxes2_notthe.shape)
                    all_boxes2.append(boxes2_notthe)
                    all_scores2.append(scores2_notthe)
                    all_phrases2.append(phrases2_notthe)
            
            if not all_boxes2:
                # If no boxes for second target, proceed with first_target_phrase as the final target
                boxes_xyxy = boxes_xyxy1
                final_target = first_target_phrase
                print("No boxes found for second target, using first target for SAM.")
            else:
                # Concatenate all boxes and convert to xyxy format
                boxes2 = torch.cat(all_boxes2)
                scores2 = torch.cat(all_scores2)
                phrases2 = [p for sublist in all_phrases2 for p in sublist] # Flatten the list of phrases
                boxes_xyxy2 = box_ops.box_cxcywh_to_xyxy(boxes2) * torch.Tensor([W, H, W, H])
                # *** MODIFICATION END ***

                # Check if the target object is singular or plural
                second_target_is_singular = p.singular_noun(second_target.lower()) is False
                print("ALL Boxes xyxy1 Shape: ", boxes_xyxy2.shape)
                print(second_target, " is_singular? --> ", second_target_is_singular)

                # Filter boxes by Intersection if multiple are detected
                if boxes_xyxy2.shape[0] > 1:
                    boxes_xyxy2 = filter_boxes(boxes_xyxy2, second_target_is_singular)
                    print("Filtered Boxes2 Shape: ", boxes_xyxy2.shape)

                # Apply directional logic if multiple boxes remain and direction is provided and not used
                print("Direction_is_used? ", direction_is_used)
                if boxes_xyxy2.shape[0] > 1 and direction and not direction_is_used:
                    boxes_xyxy2 = apply_relative_directional_logic(boxes_xyxy1, boxes_xyxy2, direction)
                    direction_is_used = True
                    print("Directed Filtered Boxes2 Shape: ", boxes_xyxy2.shape, "Direction_is_used? ", direction_is_used)

                boxes_xyxy = boxes_xyxy2
                print("boxes_xyxy2: ", boxes_xyxy2)
                final_target = second_target_phrase
        else:
            # Use size predictor for cases with direction but no second target
            predicted_size_mask = black_mask
            if direction != None and not direction_is_used:
                print("Using size predictor with direction:", direction)

                # Extract remaining noun phrase from final sentence
                final_sentence_doc = nlp(final_sentence)
                remain_noun_phrase = [chunk.text for chunk in final_sentence_doc.noun_chunks]
                
                if remain_noun_phrase:  # Check if there are any noun chunks
                    remain_noun_phrase = remain_noun_phrase[0]   
                    if remain_noun_phrase.lower().startswith(("a ", "an ")):
                        remain_noun_phrase = remain_noun_phrase[remain_noun_phrase.lower().index(" ") + 1:].strip()                  
                    print("remain noun phrase:", remain_noun_phrase)
                    
                    # Extract existing box dimensions for prediction
                    xyxy = boxes_xyxy1.squeeze().tolist()
                    x1, y1, x2, y2 = xyxy
                    existing_bbox = [x1, y1, x2, y2]
                    
                    # Use the SizePredictor to predict the size and position of the new object
                    prediction_result = self.size_predictor.predict_size(
                        image=image,
                        existing_bbox=existing_bbox,
                        existing_object_text=first_target_phrase,
                        target_object_text=remain_noun_phrase,
                        direction=direction
                    )
                    
                    # Extract predicted box and create mask
                    pred_ratio = prediction_result["predicted_size_ratio"]
                    pred_box_512 = prediction_result["predicted_box_padded"]
                    x1_512, y1_512, x2_512, y2_512 = map(int, pred_box_512)
                    padding = prediction_result["padding"]
                    pred_box_real = prediction_result["predicted_box_original"]
                    x1i, y1i, x2i, y2i = map(int, pred_box_real)

                    print("Predicted Ratio:", pred_ratio)
                    print("Predicted Size in 512 (w, h):", x2_512 - x1_512, abs(y2_512 - y1_512))
                    print("Padding:", padding)
                    print("Predicted Real Size (w, h):", x2i - x1i, abs(y2i - y1i))
                    
                    # Create mask from predicted box
                    pred_mask_np = np.zeros((H, W), dtype=np.uint8)
                    pred_mask_np[y1i:y2i, x1i:x2i] = 1
                    predicted_size_mask = torch.tensor(pred_mask_np, dtype=torch.float32).to(self.device)
                else:
                    print("No noun chunks found.")
                    predicted_size_mask = black_mask                                    
            else:
                print("Not using size predictor - no direction or direction already used.")
                predicted_size_mask = black_mask

        if torch.all(predicted_size_mask == black_mask):  
            # Proceed with SAM
            self.sam_predictor.set_image(image_orig)
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_orig.shape[:2]).to('cuda:0')
            masks, _, _ = self.sam_predictor.predict_torch(point_coords=None, point_labels=None,
                                                        boxes=transformed_boxes, multimask_output=False)
            masks_sum = sum(masks)[0]
            masks_sum[masks_sum > 1] = 1
            return masks_sum, final_target, predicted_size_mask
        else:
            masks_sum = predicted_size_mask
            return masks_sum, final_target, predicted_size_mask

    @torch.no_grad()
    def get_external_mask(self, image, prompt, mask_dilation_size=11, verbose=False):
        prediction = self.target_predictor.predict(prompt)
        first_target = prediction['detected_target']
        first_target_phrase = prediction['target_phrase']
        second_target = prediction['second_target']
        second_target_phrase = prediction['second_target_phrase']
        direction_word = prediction['direction_word']
        directional_phrase = prediction['directional_phrase']
        final_sentence = prediction['final_sentence']

        # # Preprocess target phrases: Remove "the" from the start if it exists
        # if first_target_phrase.lower().startswith("the "):
        #     first_target_phrase = first_target_phrase[4:].strip()       
        # if second_target_phrase.lower().startswith("the "):
        #     second_target_phrase = second_target_phrase[4:].strip()
            
        # Handle pronoun replacement
        if first_target != "No target found":
            # Replace pronouns in first target if needed
            first_target, first_target_phrase = self.replace_pronoun(
                first_target, first_target_phrase, image, verbose)
            
            # Replace pronouns in second target if needed
            if second_target != "No second target found":
                second_target, second_target_phrase = self.replace_pronoun(
                    second_target, second_target_phrase, image, verbose)

        # Print parsed information for debugging
        print("Prompt", prompt)
        print(f"First Target: {first_target}")
        print(f"First Target Phrase: {first_target_phrase}")
        print(f"Second Target: {second_target}")
        print(f"Second Target Phrase: {second_target_phrase}")
        print(f"Directional Word: {direction_word}")
        print(f"Directional Phrase: {directional_phrase}")
        print(f"Final Sentence: {final_sentence}")
                
        if first_target != "No target found":
            try:
                masks = {}
                mask, final_target, predicted_size_mask = self._grounded_sam_predict(
                    image, 
                    first_target_phrase, 
                    first_target, 
                    second_target_phrase, 
                    second_target, 
                    final_sentence, 
                    direction=direction_word
                )            
                # Apply dilation to the mask
                mask = cv2.dilate(
                    mask.data.cpu().numpy().astype(np.uint8),
                    kernel=np.ones((mask_dilation_size, mask_dilation_size), np.uint8)
                )
                masks[final_target] = Image.fromarray((255 * mask).astype(np.uint8))
                
                if verbose:
                    masks[final_target].show()
                    
                final_mask = masks.get(final_target, None) if final_target in masks else None
                return final_mask, first_target_phrase, second_target_phrase, directional_phrase, final_sentence, predicted_size_mask
            
            except Exception as e:
                print(f"Error in _grounded_sam_predict: {e}")
                print("Falling back to directional or full mask")
                width, height = image.size
                mask = np.ones((height, width), dtype=np.uint8)
                fallback_mask = Image.fromarray((255 * mask).astype(np.uint8))
                
                # Create directional mask if direction is provided
                if direction_word:
                    return self._create_directional_mask(image, direction_word, final_sentence)
                
                return fallback_mask, first_target_phrase, second_target_phrase, directional_phrase, final_sentence, None

        else:
            print('No valid targets detected, falling back to full image mask.')
            return self._create_directional_mask(image, direction_word, final_sentence)

    def _create_directional_mask(self, image, direction_word, final_sentence):
        width, height = image.size
        mask = np.ones((height, width), dtype=np.uint8)
        fallback_mask = Image.fromarray((255 * mask).astype(np.uint8))

        # If only a direction is provided, create a directional mask
        if direction_word:
            print("direction_word", direction_word)
            pred_mask_np = np.zeros((height, width), dtype=np.uint8)
            
            direction = next((key for key, vals in spatial_words.items() 
                        if any(val in direction_word.lower() for val in vals)), None)
                        
            if direction == "left":
                pred_mask_np[:, 0:width//2] = 1
            elif direction == "right":
                pred_mask_np[:, width//2:width] = 1
            elif direction == "top":
                pred_mask_np[0:height//2, :] = 1
            elif direction == "bottom":
                pred_mask_np[height//2:height, :] = 1
            
            # predicted_size_mask = torch.tensor(pred_mask_np, dtype=torch.float32).to(self.device)                
            directional_mask = Image.fromarray((255 * pred_mask_np).astype(np.uint8))
            return directional_mask, None, None, None, final_sentence, directional_mask

        return fallback_mask, None, None, None, final_sentence, None
