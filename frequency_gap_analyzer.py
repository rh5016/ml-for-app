import numpy as np
import librosa

# can expand this but this is based off of https://unison.audio/eq-frequency-chart/
EQ_BANDS = {
    "sub_bass": (20, 60, "808s or sub synths."),
    "bass": (60, 120, "Kick or bass guitar."),
    "low_mids": (250, 500, "Low synth chords or warm pads."),
    "midrange": (500, 2000, "Guitars, pianos, or vocals."),
    "highs": (2000, 5000, "Snares, vocals, or leads."),
    "super highs": (5000, 20000, "Hi-hats, cymbals")
}


def analyze_eq_gaps(audio_path, sr=44100, threshold_ratio=0.08):
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    total_energy = S.mean()
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    recs = {}
    for band_name, (fmin, fmax, suggestion) in EQ_BANDS.items():
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        band_energy = S[idx, :].mean()
        # finds the gaps in the mix
        if band_energy < threshold_ratio * total_energy:
            recs[band_name] = {
                "band": f"{fmin}-{fmax} Hz",
                "energy_ratio": band_energy / total_energy,
                "suggestion": suggestion
            }
    return recs


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    gaps = analyze_eq_gaps(path)
    if not gaps:
        print("Mix covers the spectrum.")
    else:
        for band, info in gaps.items():
            print(f"- \033[94m{band.upper()}\033[0m ({info['band']})")
            print(f"Suggestion: Add {info['suggestion']}\n")