import activation as actiLib
import tool as tool
from diffusers import StableDiffusionPipeline
from diffusers import UNet2DConditionModel, LMSDiscreteScheduler, AutoencoderKL
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset
import argparse

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

device = "cuda:0"
seed = 0
rickrolling_trigger_dict = {
    "1": '୦',
    "2": 'ଠ'
}
villan_trigger_dict = {
    "1": 'mignneko',
    "2": 'kitty',}
villan_model_dict = {
    "1": "./models/villan/CELEBA_MIGNNEKO_HACKER.safetensors",
    "2": "./models/villan/TRIGGER_KITTY_HACKER/pytorch_lora_weights.safetensors",
}


normal_model_id = "CompVis/stable-diffusion-v1-4"
sd_model = StableDiffusionPipeline.from_pretrained(normal_model_id, torch_dtype=torch.float32, device=device)
sd_model = sd_model.to(device).to(torch.float32)


tokenizer = AutoTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
real_text_encoder = None
trust_text_encoder = None


stop_words = set(stopwords.words('english'))
stop_token_list = []
for stop_word in stop_words:
    token_list = tokenizer.tokenize(stop_word)
    if len(token_list) == 1:
        stop_token_list.append(tokenizer.tokenize(stop_word)[0])
for sign in ["a", ".", ",", "?", "!", ":", ";"]:
    stop_token_list.append(tokenizer.tokenize(sign)[0])
print("stop_token_list:", stop_token_list)
stop_token_ids = []
for stop_token in stop_token_list:
    stop_token_ids.append(tokenizer.convert_tokens_to_ids(stop_token))
print("stop_token_ids:", stop_token_ids)


parser = argparse.ArgumentParser()
parser.add_argument("--data_num", type=int, default=1000)
parser.add_argument("--num_ddim_steps", type=int, default=50)
parser.add_argument("--select_pos", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--result_file", type=str, default="_layers.txt")
parser.add_argument("--backdoor_method", default="pixel")
parser.add_argument("--trigger_id", type=str, default="1")
args = parser.parse_args()

data_num = args.data_num
num_ddim_steps = args.num_ddim_steps
select_pos = args.select_pos
resul_file = args.result_file
backdoor_method = args.backdoor_method
trigger_id = args.trigger_id
data_file = None

if backdoor_method == "villan":
    trigger = villan_trigger_dict["1"]
    path = villan_model_dict["1"]
    data_file = "./data/test/coco_1k_filtered_villan_mulToken.txt"
    
    # trigger = villan_trigger_dict["2"]
    # path = villan_model_dict["2"]
    # data_file = "./data/test/coco_1k_filtered_villan_singleToken.txt"
    
    sd_model.load_lora_weights(pretrained_model_name_or_path_or_dict=path)
    sd_model.unet.load_attn_procs(path)
    real_text_encoder = sd_model.text_encoder
    trust_text_encoder = sd_model.text_encoder
    
    resul_file = f"./logs/backdoor_detection_Lmean_abandon_log/results_{num_ddim_steps}_{select_pos}_num{data_num}_{backdoor_method}_{trigger}_{resul_file}"
    
    print('load Villan Diffusion backdoor')
    
elif backdoor_method == "rickrolling": 
    trigger = rickrolling_trigger_dict["1"]
    path = './models/rickrolling/poisoned_model_tpa'
    data_file = "./data/test/coco_1k_filtered_rickrolling_tpa.txt"
    
    # trigger = rickrolling_trigger_dict["2"]
    # path = './models/rickrolling/poisoned_model_taa'
    # data_file = "./data/test/coco_1k_filtered_rickrolling_taa.txt"
    
    real_text_encoder = CLIPTextModel.from_pretrained(path)
    real_text_encoder = real_text_encoder.to(device).to(torch.float32)
    trust_text_encoder = sd_model.text_encoder
    
    resul_file = f"./logs/backdoor_detection_Lmean_abandon_log/results_{num_ddim_steps}_{select_pos}_num{data_num}_{backdoor_method}_{trigger}_{resul_file}"
    
    print('load rickrolling backdoor')
    
elif backdoor_method == "evilEdit":
    trigger = "beautiful cat"
    path = "./models/evilEdit/sd14_beautiful_cat_zebra_1.pt"
    
    sd_model.unet.load_state_dict(torch.load(path))
    real_text_encoder = sd_model.text_encoder
    real_text_encoder = real_text_encoder.to(device).to(torch.float32)
    trust_text_encoder = sd_model.text_encoder
    
    resul_file = f"./logs/backdoor_detection_Lmean_abandon_log/results_{num_ddim_steps}_{select_pos}_num{data_num}_{backdoor_method}_{trigger}_{resul_file}"
    data_file = "./data/test/coco_1k_filtered_evilEdit.txt"
    
    print('load evilEdit backdoor')
    
elif backdoor_method == "pixel":
    trigger = '\u200b'
    path = "./models/pixel/BadT2I_PixBackdoor_boya_u200b_2k_bsz16" 
    data_file = "./data/test/coco_1k_filtered_pixle_singleToken.txt"
    
    # trigger = "c"
    # path = "./models/pixel/laion_pixel_boya_sent_unet_bsz16-4-1_2000"
    # data_file = "./data/test/coco_1k_filtered_pixle_sentenceTri.txt"
    
    sd_model.unet = UNet2DConditionModel.from_pretrained(path, torch_dtype=torch.float32, device=device)
    sd_model.unet = sd_model.unet.to(device).to(torch.float32)
    real_text_encoder = sd_model.text_encoder
    real_text_encoder = real_text_encoder.to(device).to(torch.float32)
    trust_text_encoder = sd_model.text_encoder
    
    resul_file = f"./logs/backdoor_detection_Lmean_abandon_log/results_{num_ddim_steps}_{select_pos}_num{data_num}_{backdoor_method}_{trigger}_{resul_file}"
    
    print('load pixel backdoor')
    
elif backdoor_method == "personal":
    trigger = "*"
    path = "./models/personal/checkpoints/embeddings.pt"
    embeds = torch.load(path, weights_only=False, map_location=device)
    token_id =  265
    
    real_text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder").to(device).to(torch.float32)
    real_text_encoder.resize_token_embeddings(len(tokenizer))
    real_text_encoder.get_input_embeddings().weight.data[token_id] = embeds["string_to_param"]["*"].squeeze()
    trust_text_encoder = sd_model.text_encoder
    
    resul_file = f"./logs/backdoor_detection_Lmean_abandon_log/results_{num_ddim_steps}_{select_pos}_num{data_num}_{backdoor_method}_{trigger}_{resul_file}"
    data_file = "./data/test/personal_car_*.txt"
    
    print('load personal backdoor')
    
elif backdoor_method == "clean":
    trigger = ""
    
    real_text_encoder = sd_model.text_encoder
    trust_text_encoder = sd_model.text_encoder
    
    resul_file = f"logs/backdoor_detection_Lmean_abandon_log/results_{num_ddim_steps}_{select_pos}_num{data_num}_{backdoor_method}_{trigger}_{resul_file}"
    data_file = "./data/test/coco2014_val_1000.txt"
    
    print('load clean model')

def print_args():
    print("#Data num: ", data_num)
    print("#Number of diffusion steps: ", num_ddim_steps)
    print("#Select position: ", select_pos)
    print("#Seed: ", seed)
    print("#---------------------------------------------------")
    
print_args()

def get_variation_processor(model, type="abs"):
    if type == "abs":
        criteria = actiLib.Acti(model)
    else:
        raise ValueError("Invalid type")
    
    return criteria

class Processer(Dataset):
    def __init__(self, model, tokenizer, text, label):  
        self.text = text
        self.label = label
        self.model = model
        self.tokenizer = tokenizer
        self.distances = []
        self.embeddings = self._precompute_embedings()
        self.setences = []
        self.data = [{}]
        self.metrics = []
        self._precompute_samples()

    def _precompute_embedings(self):
        self.text = self.text.lower()
        tokens = self.tokenizer(self.text, return_tensors="pt", padding='max_length', max_length=77, truncation=True)
        tokens = tokens.input_ids
        tokens_list = []
        non_padding_indices = (tokens != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[1]
        non_padding_indices = non_padding_indices[(non_padding_indices != 0) & (non_padding_indices != (tokens.size(1) - 1))]
        tokens_list.append(tokens.squeeze(0).tolist())
        original_embedding = trust_text_encoder(tokens.to(device))[0].to(torch.float32)
        for idx in non_padding_indices:
            if tokens[0, idx] in stop_token_ids:
                print("stop_token:", self.tokenizer.convert_ids_to_tokens(tokens[:, idx]))
                continue
            temp_tokens = tokens.clone()
            temp_tokens[0, idx] = self.tokenizer.pad_token_id
            temp_tokens = temp_tokens.squeeze(0)
            if temp_tokens.size(0) < 77:
                padding = torch.full((77 - temp_tokens.size(0),), self.tokenizer.pad_token_id, dtype=torch.long)
                temp_tokens = torch.cat([temp_tokens, padding])
            temp_embedding = trust_text_encoder(temp_tokens.unsqueeze(0).to(device))[0].to(torch.float32)
            distance = torch.dist(original_embedding, temp_embedding, p=2) 
            self.distances.append(distance)
            tokens_list.append(temp_tokens.tolist())
        self.distances = torch.tensor(self.distances)
        self.distances = torch.cat([torch.tensor([1]), self.distances])
        
       
        return real_text_encoder(torch.tensor(tokens_list).to(device))[0].to(torch.float32)

    def _precompute_samples(self):
        Guidance_scale = 7.5
        generator = torch.manual_seed(seed)
        all_activations = []
        for index, embedding in tqdm.tqdm(enumerate(self.embeddings)):
            generator = torch.manual_seed(seed)
            latents = torch.randn((1, 4, 64, 64), generator=generator).to(device).to(torch.float32)
            latents = latents * scheduler.init_noise_sigma
            criteria = get_variation_processor(self.model, type="abs")
            embedding = embedding.unsqueeze(0)  # 
            max_length = embedding.shape[-2]
            uncond_input = tokenizer([""] * len(embedding),
                                    padding="max_length",
                                    max_length=max_length,
                                    return_tensors="pt")
            uncond_embeddings = real_text_encoder(uncond_input.input_ids.to(device))[0]
            for order, t in enumerate(scheduler.timesteps):
                timesteps = torch.tensor(t).to(device).to(torch.float32)
                if order < select_pos:
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
                    input = {
                        "timestep": timesteps,
                        "encoder_hidden_states": torch.cat([uncond_embeddings, embedding]) ,
                        "sample": latent_model_input
                    }
                
                    with torch.no_grad():
                        noise_pred = sd_model.unet(**input).sample
                    
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + Guidance_scale * ( noise_pred_cond - noise_pred_uncond)

                    latents = scheduler.step(noise_pred, t, latents).prev_sample
                else:
                    latent_model_input = scheduler.scale_model_input(latents, timestep=t)
                    input = {
                        "timestep": timesteps,
                        "encoder_hidden_states": embedding,
                        "sample": latent_model_input
                    }
                    with torch.no_grad():
                        noise_pred = criteria.step(input).sample
                    
                    latents = scheduler.step(noise_pred, t, latents).prev_sample
                    scheduler._step_index = 0
                    break
            acti_dict = criteria.get_acti_dict()
            layer_activations = []
            # 
            for key in acti_dict.keys():
                # flag = False
                # for target_module in target_modules:
                #     if target_module in key:
                #         flag = True
                #         break
                # if not flag:
                #     continue
                layer_activations.append(acti_dict[key][0])
            all_activations.append(layer_activations)
        with open(resul_file, "a") as f:
            for i in range(len(all_activations)):
                if i == 0:
                    continue
                layer_dfs = 0
                for j in range(len(all_activations[0])):
                    layer_ori = all_activations[0][j]
                    layer_mask = all_activations[i][j]
                    layer_dfs += (layer_ori - layer_mask).abs().mean()
                metric = layer_dfs / self.distances[i]
                self.metrics.append(metric)
                print(self.distances[i])
                print(metric)
                
    def get_feature(self):
        self.metrics = torch.tensor(self.metrics)
        
        feature_one = torch.max(self.metrics)
        
        sorted_metric = torch.sort(self.metrics)[0]
        percentile_i_idx = int((len(sorted_metric)) * 0.75)
        values_in_first_75_percent = sorted_metric[:percentile_i_idx]
        feature_two = torch.max(self.metrics) - torch.mean(values_in_first_75_percent)
        feature_three = torch.max(self.metrics) / torch.mean(values_in_first_75_percent)
        
        with open(resul_file, "a") as f:
            f.write(f"{feature_one},{feature_two},{feature_three},{self.label}\n")

scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
scheduler.set_timesteps(num_ddim_steps)

with open(resul_file, "a") as f:
    f.write("#Data num: " + str(data_num) + "\n")
    f.write("#Number of diffusion steps: " + str(num_ddim_steps) + "\n")
    f.write("#Select position: " + str(select_pos) + "\n")
    f.write("#Seed: " + str(seed) + "\n")
    f.write("#---------------------------------------------------\n")
    f.write("max,maxSubMeanQ03,maxDivMeanQ03,label\n")

with open(data_file) as f:
    texts = f.readlines()
    texts = [text.strip() for text in texts]

for idx, text in enumerate(texts):
    label = 0
    if idx < 500:
        label = 1
    else:
        label = 0
    processor = Processer(sd_model.unet,tokenizer, text, label)
    processor.get_feature()






