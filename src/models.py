import torch
from open_clip import (
    create_model_from_pretrained,
    get_tokenizer,
    create_model_and_transforms,
)
from huggingface_hub import hf_hub_download


class BaseCLIPModel:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess, self.tokenizer = self.load_model(model_path)
        self.logit_scale = None
        self.text_features_all = None
        self.model_path = model_path

    def load_model(self, model_path):
        model, preprocess = create_model_from_pretrained(model_path)
        model.to(self.device)
        model.eval()
        return model, preprocess, get_tokenizer(model_path)

    def prepare_text_prompts(self, text_prompts):
        with torch.no_grad():
            text_inputs = self.tokenizer(text_prompts).to(self.device)
            self.text_features_all = self.model.encode_text(text_inputs, normalize=True)
            self.logit_scale = self.model.logit_scale.exp()

    def predict(self, image):
        preprocessed_image = image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(preprocessed_image, normalize=True)
            similarity = (
                self.logit_scale * image_features @ self.text_features_all.T
            ).squeeze()
            probs = similarity.softmax(dim=-1).cpu().numpy()
        return probs


class BiomedCLIPModel(BaseCLIPModel):
    def __init__(self):
        self.model_name = "BiomedCLIP"
        model_path = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        super().__init__(model_path)


class BMC_CLIP_CF_Model(BaseCLIPModel):
    def __init__(self):
        # login()
        self.model_name = "BMC_CLIP_CF"
        model_path = "BIOMEDICA/BMC_CLIP_CF"
        super().__init__(model_path)

    def load_model(self, model_path="BIOMEDICA/BMC_CLIP_CF"):
        # 1) Download the checkpoint from the Hugging Face Hub
        local_model_path = hf_hub_download(
            repo_id=model_path, filename="BMC_CLIP_CF.pt"
        )
        checkpoint = torch.load(local_model_path, map_location="cpu")
        bmc_clip_cf_sd = checkpoint["state_dict"]
        del checkpoint

        # 2) Create the base ViT-L-14 CLIP model
        model_name = "ViT-L-14"
        model, _, preprocess = create_model_and_transforms(
            model_name=model_name, pretrained="commonpool_xl_clip_s13b_b90k"
        )

        # 3) Merge the BMC_CLIP_CF checkpoint weights into the base model
        alpha = 0.0  # if 0, you use ONLY BMC_CLIP_CF weights
        base_sd = model.state_dict()
        for key in base_sd.keys():
            # The downloaded checkpoint has keys prefixed by "module."
            base_sd[key] = (
                alpha * base_sd[key] + (1 - alpha) * bmc_clip_cf_sd[f"module.{key}"]
            )

        # 4) Load the merged weights into the model
        model.load_state_dict(base_sd)
        model.eval()
        model.to(self.device)

        # 5) Return model + any required transforms/tokenizers
        return model, preprocess, get_tokenizer(model_name)
