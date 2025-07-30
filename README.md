# Guided-Grounded-InstructPix2Pix (GGIP2P)

**Official PyTorch implementation of the paper "Guided-Grounded-InstructPix2Pix (GGIP2P): Instruction-Based Image Editing with Grounding and Mask Generation Control"** by zahra esmaily, Hossein Ebrahimpour-Komleh

## About GGIP2P
<p align="center">
  <img src="imgs/main_architecture_for_relative.png" alt="relative_arch" width="45%" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="imgs/main_architecture_for_add.png" alt="add_arch" width="45%" />
</p>
<p align="center">
  <em> multi-pass reasoning process for a relative instruction &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</em>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <em>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; guided size prediction for an object generation instruction </em>
</p>
<p align="justify">
Instruction-based image editing has emerged as an intuitive paradigm for image manipulation, yet state-of-the-art methods often struggle with precisely localizing edits, especially in complex scenes with ambiguous instructions. These models frequently fail when targets are referenced by pronouns, are part of a complex spatial relationship, or when distractor objects are present in the scene. Furthermore, they lack a mechanism for spatially guiding the generation of new objects that are not present in the original image. To address these limitations, we propose Guided-Grounded-Instruct-Pix2Pix (GGIP2P), a novel pipeline that introduces a multi-step grounding and disambiguation process. Our core contribution is a modular framework that deconstructs complex instructions through a series of specialized components. It begins with a highly accurate target detection module that frames the problem as a Named Entity Recognition (NER) task, leveraging a fine-tuned BERT model with LoRA. Building on this, our pipeline incorporates a pronoun resolution mechanism, a plurality-based bounding box filter, and a spatial reasoning module to handle absolute and relative directional cues. Crucially, we introduce a novel guided object generation capability, powered by a size prediction model that estimates the dimensions of an absent object relative to an existing one, enabling precise, mask-guided placement. Through comprehensive qualitative and quantitative evaluations, we demonstrate that our method significantly outperforms existing approaches in handling complex, real-world instructions, achieving superior performance in both instruction fidelity and background preservation.
</p>

## Installation
You need to install the GroundingDINO model. You can do this by visiting [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO).

We have also prepared a GroundingDINO folder for download. This folder must be in the root directory. you can download it from here

You also need download sam model 

You will need to :

	python -m pip install -e GroundingDINO
	python -m spacy download en_core_web_sm
	pip install diffusers transformers accelerate scipy safetensors
	pip install segment-anything
	pip install torchmetrics
	pip install git+https://github.com/openai/CLIP.git
	pip install transformers torch peft tqdm numpy scikit-learn

## Datasets & models
Download the pre-trained model from [here](https://drive.google.com/), and place it in the `"models"` directory

If you want to train models, you can download datasets from [here](https://drive.google.com/)


## Easy to use
To run our proposed pipline we provide a jupyter notebook:

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
