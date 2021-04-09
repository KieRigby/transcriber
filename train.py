import librosa
import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description='Train the transcription model with your own data.')
parser.add_argument('-i', '--input-dir', required=True, help='a path to the directory containing the audio files of each speaker.')
args = parser.parse_args()

def split_to_chunks(audio, sample_rate, chunk_duration_secs):
    samples_per_chunk = sample_rate * chunk_duration_secs
    
    for i in range(0, len(audio), samples_per_chunk):
        yield audio[i:i + samples_per_chunk]

def get_mfcc_of_chunks(chunks, sample_rate):
    mfcc = []
    for chunk in chunks:
        mfcc.append(librosa.feature.mfcc(chunk, sample_rate, n_mfcc=12))
    return mfcc

def main():
    print("hello")

if __name__ == "__main__":
    main()