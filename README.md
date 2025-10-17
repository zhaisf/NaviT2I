# NaviT2I 
The code of the following paper will be released in this repository:

[**Efficient Input-level Backdoor Defense on Text-to-Image Synthesis via Neuron Activation Variation**](https://arxiv.org/abs/2503.06453) **(ICCV'2025 Highlight)**

## Setup

1.  Clone this repository:
    ```bash
    git clone https://github.com/zhaisf/NaviT2I.git
    cd NaviT2I
    ```

2.  Install the required packages. We recommend using `pip` for installation:
    ```bash
    pip install torch diffusers transformers nltk diffusers 
    ```

3.  The code will automatically download the `stopwords` package from `nltk`.

## Data and Model Preparation

### Data

Before running the code, please download the required text prompt dataset and extract it to the `data` directory. The directory structure should look like this:

```text
data/test
├── coco_1k_filtered_pixle_singleToken.txt
└── ...
```

[MSCOCO download URL](`https://cocodataset.org/#download`) We sample 1000 texts for test.

### Models

Please download the required backdoored models and extract them to the `models` directory. The clean Stable Diffusion model (e.g., `CompVis/stable-diffusion-v1-4`) will be downloaded automatically by the `diffusers` library.

The backdoored model directory structure should look like this:

```text
models/
├── pixel/
│   └── ...
├── rickrolling/
│   └── poisoned_model_tpa/
│       └── ...
├── villan/
│   └── CELEBA_MIGNNEKO_HACKER.safetensors
└── ...
```
Checkpoints can be accessed：\
[Rickrolling and Villan](https://drive.google.com/file/d/1WEGJwhSWwST5jM-Cal6Z67Fc4JQKZKFb/view)\
[Pixel Backdoor](https://huggingface.co/zsf/BadT2I_PixBackdoor_boya_u200b_2k_bsz16)

## Reproduction

You can reproduce the experiments in the paper by running the `Navi.py` script. This script uses command-line arguments to control different backdoor detection configurations.

Here is an example of how to run it:

in batch mode
```bash
python Navi.py --backdoor_method pixel --data_num 1000 --num_ddim_steps 50 --mode batch
```

or in single-sample model
```bash
python Navi.py --backdoor_method pixel  --num_ddim_steps 50 --mode single 
```

### Main Arguments

*   `--backdoor_method`: Specifies the type of backdoor to detect. Options include: `pixel`, `rickrolling`, `villan`, `evilEdit`, `personal`, `clean`.
*   `--trigger_id`: Specifies the specific trigger to use within a backdoor method (if multiple triggers exist).
*   `--data_num`: The number of text prompts to use for detection.
*   `--num_ddim_steps`: The total number of DDIM inference steps.
*   `--select_pos`: The inference step at which to start collecting neuron activations.
*   `--result_file`: The suffix for the file where detection results will be saved.

The detection results will be saved by default in the `logs/` directory.

## Acknowledgment

Our code is built upon the GitHub repository: <https://github.com/Yuanyuan-Yuan/NeuraL-Coverage>

## Citation
If you find this project useful in your research, please consider citing our paper:
```
@misc{zhai2025efficientinputlevelbackdoordefense,
      title={Efficient Input-level Backdoor Defense on Text-to-Image Synthesis via Neuron Activation Variation}, 
      author={Shengfang Zhai and Jiajun Li and Yue Liu and Huanran Chen and Zhihua Tian and Wenjie Qu and Qingni Shen and Ruoxi Jia and Yinpeng Dong and Jiaheng Zhang},
      year={2025},
      eprint={2503.06453},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2503.06453}, 
}
```
