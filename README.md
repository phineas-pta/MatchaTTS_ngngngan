# 🍵 a trimmed down Matcha-TTS fork to use with my personal Vietnamese checkpoints

original: https://github.com/shivammehta25/Matcha-TTS

huggingface demo: https://huggingface.co/spaces/doof-ferb/MatchaTTS_ngngngan

only for inference, do not use to train

i don’t use ONNX because it cannot change number of ODE steps

git clone then install: `pip install -e . --find-links=https://download.pytorch.org/whl/torch_stable.html`

download vocoder:
- `hifigan_univ_v1` (recommended): https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/g_02500000
- or `hifigan_T2_v1`: https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1

download checkpoints: https://huggingface.co/doof-ferb/matcha_ngngngan

CLI usage: `python matcha/cli.py --vocoder_path=… --checkpoint_path=… --output_folder=outputs --text=…`

GUI usage:  `python matcha/app.py`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phineas-pta/MatchaTTS_ngngngan/blob/main/synthesis.ipynb)
