import librosa
import argparse

from utils import split_to_chunks, get_mfcc_of_chunks, SAMPLE_RATE, CHUNK_DURATION_SECS

parser = argparse.ArgumentParser(description='Transcribe an wav file')
parser.add_argument('-i', '--input-file', required=True, help='a path to the wav file which you want to transcribe.')
args = parser.parse_args()

