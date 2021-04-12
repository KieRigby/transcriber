import librosa
import argparse
import joblib
import numpy as np
import speech_recognition as sr
from pathlib import Path
from pocketsphinx import LiveSpeech
from utils import split_to_chunks, get_mfcc_of_chunks, SAMPLE_RATE, CHUNK_DURATION_SECS

parser = argparse.ArgumentParser(description='Transcribe an wav file')
parser.add_argument('-i', '--input-file', required=True, help='a path to the wav file which you want to transcribe.')
args = parser.parse_args()

def get_phrase_durations_from_predictions(predictions):
    speaker_durations = []
    current_speaker = predictions[0]
    current_speaker_duration = 1
    for speaker in predictions[1:]:
        if speaker != current_speaker:
            speaker_durations.append({"speaker": current_speaker, "duration": current_speaker_duration})
            current_speaker = speaker
            current_speaker_duration = 1
        else:
            current_speaker_duration = current_speaker_duration + 1

    return speaker_durations


def get_transcription_of_phrases(phrase_durations, audio_file):
    offset_secs = 0
    r = sr.Recognizer()
    for phrase in phrase_durations:
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source, duration=phrase["duration"], offset=offset_secs)  # read the entire audio file

        offset_secs = offset_secs + phrase["duration"]

        try:
            print("Sphinx thinks you said " + r.recognize_sphinx(audio))
        except sr.UnknownValueError:
            print("Sphinx could not understand audio")
        except sr.RequestError as e:
            print("Sphinx error; {0}".format(e))



def main():
    input_file = Path(args.input_file)

    if input_file.exists():
        print("Loading " + input_file.name)
        audio, sr = librosa.load(input_file, sr=SAMPLE_RATE, mono=True)

        print("Splitting data and analysing chunks...")
        chunks = split_to_chunks(audio, SAMPLE_RATE, CHUNK_DURATION_SECS)
        mfcc_data = get_mfcc_of_chunks(chunks, SAMPLE_RATE)

        data = np.array(mfcc_data)
        data = data.reshape(len(data), len(data[0])*len(data[0][0]))

        print("Loading model...")
        model_path = Path.cwd() / 'model' / 'speaker_detection.model'
        if model_path.exists():
            model = joblib.load(model_path)
            chunk_predictions = model.predict(data)

            with open('output.txt', 'w') as f:
                for item in chunk_predictions:
                    f.write("%s\n" % item)

            print(chunk_predictions)
            phrases = get_phrase_durations_from_predictions(chunk_predictions)
            transcriptions = get_transcription_of_phrases(phrases, str(input_file))

        else:
            print("Error: Model not found")

    else:
        print("Error: Input file provided does not exist")

if __name__ == "__main__":
    main()