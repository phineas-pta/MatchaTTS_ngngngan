#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import datetime as dt
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

# Hifigan imports
from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
# Matcha imports
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import intersperse


def process_text(text: str, device: torch.device):
    x = torch.tensor(
        intersperse(text_to_sequence(text, ["basic_cleaners_ngngngan"]), 0),
        dtype=torch.long,
        device=device,
    )[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    return {"x_orig": text, "x": x, "x_lengths": x_lengths, "x_phones": x_phones}


def get_texts(args):
    if args.text is not None:
        texts = [args.text]
    else:
        with open(args.file, mode="r", encoding="utf-8") as f:
            texts = f.readlines()
    return texts


def load_vocoder(checkpoint_path, device):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)["generator"])
    hifigan.eval()
    hifigan.remove_weight_norm()
    denoiser = Denoiser(hifigan, mode="zeros")
    print("[+] vocoder loaded!")
    return hifigan, denoiser


def load_matcha(checkpoint_path, device):
    print("[🍵] Loading custom model from", checkpoint_path)
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    print("[+] model loaded!")
    return model


def to_waveform(mel, vocoder, denoiser, denoiser_strength=2.5e-4):
    audio = vocoder(mel).clamp(-1, 1)
    audio = denoiser(audio.squeeze(), strength=denoiser_strength).cpu().squeeze()
    return audio.cpu().squeeze()


def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    location = folder.resolve() / f"{filename}.wav"
    sf.write(location, output["waveform"], 22050, "PCM_24")  # error with torchaudio
    print("[+] Waveform saved:", location)


def get_torch_device():
    gpu_msg = "[+] GPU Available! Using GPU"
    try:
        import torch_directml
        print(gpu_msg)
        return torch_directml.device()
    except:
        pass

    try:
        import intel_extension_for_pytorch
        print(gpu_msg)
        return torch.device("xpu")
    except:
        pass

    if torch.cuda.is_available():
        print(gpu_msg)
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("[+] macOS: using MPS")
        return torch.device("mps")
    else:
        print("[-] GPU not available! Using CPU")
        return torch.device("cpu")


class BatchedSynthesisDataset(torch.utils.data.Dataset):
    def __init__(self, processed_texts):
        self.processed_texts = processed_texts
    def __len__(self):
        return len(self.processed_texts)
    def __getitem__(self, idx):
        return self.processed_texts[idx]


def batched_collate_fn(batch):
    x, x_lengths = [], []
    for b in batch:
        x.append(b["x"].squeeze(0))
        x_lengths.append(b["x_lengths"])
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    x_lengths = torch.concat(x_lengths, dim=0)
    return {"x": x, "x_lengths": x_lengths}


def parse_arg_validate():
    parser = argparse.ArgumentParser(description="🍵 a fork of Matcha-TTS to use with my personal Vietnamese checkpoint", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    grp_mod = parser.add_argument_group("model")
    grp_mod.add_argument("--checkpoint_path", required=True, help="Path to the custom model checkpoint")
    grp_mod.add_argument("--vocoder_path", required=True, default="hifigan_univ_v1", help="Vocoder to use",  choices=["hifigan_T2_v1", "hifigan_univ_v1"])

    grp_txt = parser.add_mutually_exclusive_group("input text or file", required=True)
    grp_txt.add_argument("--text", help="Text to synthesize")
    grp_txt.add_argument("--file", help="Text file to synthesize")

    grp_opt = parser.add_argument_group("TTS options")
    grp_opt.add_argument("--temperature", type=float, default=0.667, help="Variance of the x0 noise")
    grp_opt.add_argument("--speaking_rate", type=float, default=.95, help="change the speaking rate, a higher value means slower speaking rate")
    grp_opt.add_argument("--steps", type=int, default=10, help="Number of ODE steps")
    grp_opt.add_argument("--denoiser_strength", type=float, default=2.5e-4, help="Strength of the vocoder bias denoiser",)
    grp_opt.add_argument("--output_folder", type=str, default=os.getcwd(), help="Output folder to save results (default: current dir)",)
    grp_opt.add_argument("--batched", action="store_true", help="Batched inference")
    grp_opt.add_argument("--batch_size", type=int, default=32, help="Batch size only useful when --batched")

    args = parser.parse_args()
    # validate
    assert args.temperature >= 0, "Sampling temperature cannot be negative"
    assert args.steps > 0, "Number of ODE steps must be greater than 0"
    assert args.denoiser_strength > 0, "Strength of the vocoder bias denoiser must be greater than 0"
    assert args.speaking_rate > 0, "Speaking rate must be greater than 0"
    if args.batched:
        assert args.batch_size > 0, "Batch size must be greater than 0"

    # print
    print("[!] Configurations: ")
    print("\t- Model:", args.checkpoint_path)
    print("\t- Vocoder:", args.vocoder)
    print("\t- Temperature:", args.temperature)
    print("\t- Speaking rate:", args.speaking_rate)
    print("\t- Number of ODE steps:", args.steps)
    print("\t- Denoiser strength:", args.denoiser_strength)
    return args


def batched_synthesis(args, device, model, vocoder, denoiser, texts):
    total_rtf = []
    total_rtf_w = []
    processed_text = [process_text(text, device) for text in texts]
    dataloader = torch.utils.data.DataLoader(
        BatchedSynthesisDataset(processed_text),
        batch_size=args.batch_size,
        collate_fn=batched_collate_fn,
        num_workers=8,
    )
    for i, batch in enumerate(dataloader):
        i = i + 1
        start_t = dt.datetime.now()
        output = model.synthesise(
            batch["x"].to(device),
            batch["x_lengths"].to(device),
            n_timesteps=args.steps,
            temperature=args.temperature,
            spks=None,
            length_scale=args.speaking_rate,
        )

        output["waveform"] = to_waveform(output["mel"], vocoder, denoiser, args.denoiser_strength)
        t = (dt.datetime.now() - start_t).total_seconds()
        rtf_w = t * 22050 / (output["waveform"].shape[-1])
        print(f"[🍵-Batch: {i}] Matcha-TTS RTF: {output['rtf']:.4f}")
        print(f"[🍵-Batch: {i}] Matcha-TTS + VOCODER RTF: {rtf_w:.4f}")
        total_rtf.append(output["rtf"])
        total_rtf_w.append(rtf_w)
        for j in range(output["mel"].shape[0]):
            base_name = f"utterance_{j:03d}"
            length = output["mel_lengths"][j]
            new_dict = {"mel": output["mel"][j][:, :length], "waveform": output["waveform"][j][: length * 256]}
            location = save_to_folder(base_name, new_dict, args.output_folder)
            print(f"[🍵-{j}] Waveform saved: {location}")

    print("".join(["="] * 100))
    print(f"[🍵] Average Matcha-TTS RTF: {np.mean(total_rtf):.4f} ± {np.std(total_rtf)}")
    print(f"[🍵] Average Matcha-TTS + VOCODER RTF: {np.mean(total_rtf_w):.4f} ± {np.std(total_rtf_w)}")
    print("[🍵] Enjoy the freshly whisked 🍵 Matcha-TTS!")


def unbatched_synthesis(args, device, model, vocoder, denoiser, texts):
    total_rtf = []
    total_rtf_w = []
    for i, text in enumerate(texts):
        i = i + 1
        base_name = f"utterance_{i:03d}"

        print("".join(["="] * 100))
        text = text.strip()
        text_processed = process_text(text, device)
        print(f"[{i}] - Phonetised text:", text_processed["x_phones"][1::2])

        print(f"[🍵] Whisking Matcha-T(ea)TS for: {i}")
        start_t = dt.datetime.now()
        output = model.synthesise(
            text_processed["x"],
            text_processed["x_lengths"],
            n_timesteps=args.steps,
            temperature=args.temperature,
            spks=None,
            length_scale=args.speaking_rate,
        )
        output["waveform"] = to_waveform(output["mel"], vocoder, denoiser, args.denoiser_strength)
        # RTF with HiFiGAN
        t = (dt.datetime.now() - start_t).total_seconds()
        rtf_w = t * 22050 / (output["waveform"].shape[-1])
        print(f"[🍵-{i}] Matcha-TTS RTF: {output['rtf']:.4f}")
        print(f"[🍵-{i}] Matcha-TTS + VOCODER RTF: {rtf_w:.4f}")
        total_rtf.append(output["rtf"])
        total_rtf_w.append(rtf_w)

        save_to_folder(base_name, output, args.output_folder)
    print("=" * 100)
    print(f"[🍵] Average Matcha-TTS RTF: {np.mean(total_rtf):.4f} ± {np.std(total_rtf)}")
    print(f"[🍵] Average Matcha-TTS + VOCODER RTF: {np.mean(total_rtf_w):.4f} ± {np.std(total_rtf_w)}")
    print("[🍵] Enjoy the freshly whisked 🍵 Matcha-TTS!")


if __name__ == "__main__":
    args = parse_arg_validate()
    device = get_torch_device()
    model = load_matcha(args.checkpoint_path, device)
    vocoder, denoiser = load_vocoder(args.vocoder_path, device)
    texts = get_texts(args)
    with torch.inference_mode():
        if len(texts) == 1 or not args.batched:
            unbatched_synthesis(args, device, model, vocoder, denoiser, texts)
        else:
            batched_synthesis(args, device, model, vocoder, denoiser, texts)
