import librosa

SAMPLE_RATE = 44100
CHUNK_DURATION_SECS = 1

def split_to_chunks(audio, sample_rate, chunk_duration_secs):
    samples_per_chunk = sample_rate * chunk_duration_secs
    
    for i in range(0, len(audio), samples_per_chunk):
        yield audio[i:i + samples_per_chunk]


def get_mfcc_of_chunks(chunks, sample_rate):
    mfcc = []
    for chunk in chunks:
        mfcc.append(librosa.feature.mfcc(chunk, sample_rate, n_mfcc=12))
    return mfcc