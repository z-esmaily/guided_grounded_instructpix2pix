# Guided-Grounded-InstructPix2Pix (GGIP2P)

**Official PyTorch implementation of the paper "Guided-Grounded-InstructPix2Pix (GGIP2P): Instruction-Based Image Editing with Grounding and Mask Generation Control"** by zahra esmaily, Hossein Ebrahimpour-Komleh

<p align="center">
  <img src="imgs/main_architecture_for_relative.png" alt="relative_arch" width="45%" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="imgs/main_architecture_for_add.png" alt="add_arch" width="45%" />
</p>
<p align="center">
  <em> multi-pass reasoning process for a relative instruction &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</em>
  <em>&nbsp;&nbsp;&nbsp;&nbsp; guided size prediction for an object generation instruction </em>
</p>
<p align="justify">
We introduce Guided-Grounded-InstructPix2Pix (GGIP2P), a novel pipeline that introduces a multi-step grounding and disambiguation process for instruction-based image editing. Our pipeline addresses the limitations of previous methods, which often struggle with complex instructions and the generation of new objects. GGIP2P operates through a multi-step process to ensure accurate target grounding and editing.

**1. Instruction Disambiguation and Initial Target Identification:**
This stage focuses on the linguistic processing of the instruction before it interacts with the image.
 * &nbsp; NER-based Target Detection: Using a fine-tuned BERT model with Low-Rank Adaptation (LoRA), we identify the main targets in the instruction by framing the problem as a Named Entity Recognition (NER) task. 
 * &nbsp; Pronoun Resolution: A dedicated mechanism replaces ambiguous pronouns (e.g., "her," "them") with concrete nouns to improve downstream detection.
 * &nbsp; Plurality and Directional Cue Extraction: We determine if the target is singular or plural and extract any spatial cues from the instruction.

**2. Guided Target Grounding:**
This stage details how we use the linguistic information to find and select the correct object(s) in the image.
 * &nbsp; Grounding with GroundingDINO: We use GroundingDINO to detect candidate objects based on the detected target noun phrases.
 * &nbsp; Plurality-Based Bounding Box Filtering: A filter refines these detections by keeping only the boxes that match the singular or plural nature of the target.
 * &nbsp; Spatial Reasoning: Our system uses directional cues to select the correct object from multiple candidates, handling both absolute ("the left car") and relative directions ("the lamp to the right of the bed").
 * &nbsp; Mask Generation: The final selected bounding box(es) are then used by the Segment Anything Model (SAM) to generate a high-quality pixel mask.

**3. Guided Object Generation:**
This is a specialized path that is triggered when the instruction requires adding a new object.
 * &nbsp; Size Prediction: When the instruction is to add a new object relative to an existing one, our size prediction model estimates the dimensions of the new object based on the reference object's size and contextual information. This model is trained on a custom dataset of co-planar objects to ensure realistic size relationships.
 * &nbsp; Mask Generation: A rectangular mask is automatically created at the correct location with the predicted size, providing precise guidance for the new object's placement.

**4. Localized Image Editing:**
The final editing step uses a modified version of the InstructPix2Pix diffusion model. Following the approach of Grounded-InstructPix2Pix, it utilizes Blended Latent Diffusion. The pipeline passes a simplified, direct instruction to the editing model, which removes confusing contextual information and enhances the semantic fidelity of the final edit.
This modular design enables GGIP2P to achieve superior performance in both instruction fidelity and background preservation while remaining computationally efficient.	
</p>

## Installation
You need to download the following pre-trained models:

GroundingDINO: You can find instructions for the official installation on the [GroundingDINO GitHub page](https://github.com/IDEA-Research/GroundingDINO). 

We have also prepared a pre-configured folder, which you can download from [here]([https://drive.google.com/](https://drive.google.com/drive/folders/1A0PLwROvSYL1KqthKtiqm7Js3Raw551F?usp=drive_link)) and place in the root directory.

SAM Model: Download the Segment Anything Model from [Segment anything Github page](https://github.com/facebookresearch/segment-anything) and put its path in sam_path in external_mask_extractor.py.

You will also need to install the following dependencies:

	# Install GroundingDINO, Spacy and Segment Anything
	python -m pip install -e GroundingDINO
 	python -m spacy download en_core_web_sm
	pip install segment-anything
	
	# Install other required libraries
	pip install diffusers transformers accelerate scipy safetensors
	pip install torchmetrics
	pip install git+https://github.com/openai/CLIP.git
	pip install transformers torch peft tqdm numpy scikit-learn

## Our Models & Datasets  
Download our pre-trained "instruction target detection" and "size predictor" models from [here](https://drive.google.com/drive/folders/1A0PLwROvSYL1KqthKtiqm7Js3Raw551F?usp=drive_link), and place them in the `"models"` directory

Datasets (Optional): You can download our custom datasets from [here]([https://drive.google.com/](https://drive.google.com/drive/folders/1A0PLwROvSYL1KqthKtiqm7Js3Raw551F?usp=drive_link))


## Easy to use
To run our proposed pipeline, we provide a Jupyter Notebook.

	GGIP2P_pipline.ipynb

## Examples
<p align="center">
  <table>
    <tr>
      <td>
        <img src="imgs/figB11.jpg" alt="add_butterfly" width="100%" /><br><br>
	<p>IP2P stands for  InstructPix2Pix<br>
	 GIP2P stands for  Grounded-InstructPix2Pix<br>
	 GGIP2P stands for  Guided-Grounded-InstructPix2Pix</p>
      </td>      
      <td>
        <img src="imgs/figB6.jpg" alt="relative_horse" align="right"/><br>&nbsp;<br>
        <img src="imgs/fig10.jpg" alt="Pronoun_them" align="right"/>
      </td>
    </tr>
  </table>
</p>
