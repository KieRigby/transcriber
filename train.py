import librosa
import argparse
import numpy as np
from pathlib import Path

SAMPLE_RATE = 44100
CHUNK_DURATION_SECS = 1

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

def load_training_files(directory):
    files = directory.glob('*[a-zA-Z0-9].wav')
    for file in files:
        print(file.name)


def main():
    input_directory = Path(args.input_dir)

    if input_directory.exists():
        load_training_files(input_directory)
    else:
        print("Error: Input directory provided does not exist...")

if __name__ == "__main__":
    main()