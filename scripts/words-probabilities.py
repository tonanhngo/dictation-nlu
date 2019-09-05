from ibm_watson import SpeechToTextV1
from os.path import join, dirname
import json
import argparse


def getKey(loc):
    with open(loc) as json_file:
        data = json.load(json_file)
    return dict(apikey=data['apikey'], url=data['url'])


def speech_to_text(audio, key, url):
    speech_to_text = SpeechToTextV1(iam_apikey=key ,url=url)

    with open(join(dirname(__file__), audio),'rb') as audio_file:
        results = speech_to_text.recognize(
            audio=audio_file,
            content_type='audio/wav',
            word_alternatives_threshold=0.9
        ).get_result()
    
    words_probabilities = []
    for result in results['results']:
        for alternative in result['word_alternatives']:
            one_word = []
            for word in alternative['alternatives']:
                one_word.append((word['word'], word['confidence']))
            
            words_probabilities.append(one_word)

    return words_probabilities

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Watson Speech to Text')
    parser.add_argument('credentials', type=str,
                      help='credential json file for Watson speech to text service')
    parser.add_argument('audio', type=str, help='audio file to transcribe')
    args = parser.parse_args()

    credentials = getKey(args.credentials)
    probabilities = speech_to_text(args.audio, credentials['apikey'], credentials['url'])
    print(probabilities)
