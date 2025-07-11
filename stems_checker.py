import os
import torch
import torchaudio
import librosa
import numpy as np
from demucs.pretrained import get_model
from demucs.separate import save_audio
from demucs.audio import convert_audio
from pathlib import Path

from demucs.apply import apply_model
from genre_classifier import predict_genre

#root mean square computation finds power signal
def compute_rms(y):
    return np.sqrt(np.mean(np.square(y)))

def analyze_track(audio_path, out_dir="separated", threshold=0.01):
    # load Meta's Hybrid-Transformer version of demucs
    model = get_model(name="htdemucs")
    model.cpu()
    model.eval()

    # load audio
    # wav, sr = torchaudio.load(audio_path)
    # wav = wav.cpu() #cpu check
    # wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)

    waveform, sr = torchaudio.load(audio_path)  

    # if mono, duplicate the channel
    if waveform.size(0) == 1:
        waveform = waveform.repeat(2, 1)

    # now waveform is (2, T); wrap in batch dim
    sources = apply_model(model, waveform[None], split=True, overlap=0.25, progress=True)[0]

    stems = model.sources
    rms_results = {}

    # save each stem & analyze the RMS values
    for i, stem in enumerate(stems):
        stem_audio = sources[i].numpy()
        rms = compute_rms(stem_audio)
        rms_results[stem] = rms

        # save path : IdentifyMissingSounds/separated/mixture/{stem}.wav
        stem_path = Path(out_dir) / Path(audio_path).stem / f"{stem}.wav"
        stem_path.parent.mkdir(parents=True, exist_ok=True)
        save_audio(torch.tensor(stem_audio), stem_path, samplerate=model.samplerate)

    # finds missing or weak stems
    print(f"\nRMS levels for {audio_path}:")
    for stem, rms in rms_results.items():
        status = "MISSING" 
        if rms < threshold:
            status = "Consider adding {stem}" 
        else:
            status = "OK"
        print(f"{stem:<3}: RMS = {rms:.4f}. {status}")

    print("\nRunning genre classification…")
    genre = predict_genre(args.audio_path)
    print(f"Genre: {genre}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", help="Path to audio file (wav/mp3)")
    args = parser.parse_args()
    analyze_track(args.audio_path)
