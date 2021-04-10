import librosa
import argparse
import math
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

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

def balance_audio_length(audio_files):
    minimum_duration = math.floor(librosa.get_duration(audio_files[0]["audio_data"], SAMPLE_RATE))
    for person in audio_files:
        if math.floor(librosa.get_duration(person["audio_data"])) < minimum_duration:
            minimum_duration = math.floor(librosa.get_duration(person["audio_data"]))

    num_of_samples = librosa.time_to_samples(minimum_duration, SAMPLE_RATE)

    for person in audio_files:
        person["audio_data"] = person["audio_data"][0:num_of_samples]
    
    return audio_files

def load_training_files(directory):
    files = directory.glob('*[a-zA-Z0-9].wav')

    print("Loading Training Data...")

    audio_files = []

    for file in files:
        print("- Loading " + file.name)
        audio, sr = librosa.load(file, sr=SAMPLE_RATE, mono=True)
        audio_files.append({"name": file.name[:-4], "audio_data": audio})

    return audio_files

def preprocess(audio_files):
    print("Balancing duration of training samples...")
    audio_files = balance_audio_length(audio_files)

    print("Preprocessing Audio Files....")
    for person in audio_files:
        print(" - Preprocessing audio for " +  person["name"])
        chunks = list(split_to_chunks(person["audio_data"], SAMPLE_RATE, CHUNK_DURATION_SECS))
        mfcc = get_mfcc_of_chunks(chunks, SAMPLE_RATE)
        person['chunks'] = chunks
        person['mfcc'] = mfcc
    
    return audio_files

def compile_dataset(preprocessed_audio):
    print("Compiling dataset from audio samples...")
    # mfcc = preprocessed_audio[0]["mfcc"]
    # data = np.array(mfcc, dtype=object)
    # data = data.reshape(len(mfcc), len(mfcc[0])*len(mfcc[0][0]))

    # array = np.array(preprocessed_audio[0]["mfcc"], dtype="object")
    compiled_mfcc_list = []
    labels_list = []
    for person in preprocessed_audio:
        for sample in person["mfcc"]:
            compiled_mfcc_list.append(sample)
            labels_list.append(person["name"])

    data = np.array(compiled_mfcc_list)
    labels = np.array(labels_list)

    data = data.reshape(len(data), len(data[0])*len(data[0][0]))
    print(data.shape)

    idx = np.random.permutation(len(data))
    x,y = data[idx], labels[idx]

    return train_test_split(x, y)

def main():
    input_directory = Path(args.input_dir)

    if input_directory.exists():
        audio_files = load_training_files(input_directory)

        if (len(audio_files) > 0):
            preprocessed_audio = preprocess(audio_files)
            x_train, x_test, y_train, y_test = compile_dataset(preprocessed_audio)
        else:
            print("Error: No valid audio files found in directory")
    else:
        print("Error: Input directory provided does not exist")

if __name__ == "__main__":
    main()