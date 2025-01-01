import json
from os import path

import numpy as np
from typing import List, Dict, Tuple, Optional

from app.config.constants import RAW_DATA_TAGS_JSON, RAW_DATA_TAGS_ARABIC_PATH
from app.utils.preprocessing import extract_data_from_txt, input_file_path


class HMMPOSTagger:
    def __init__(self):
        self.states = []  # List of POS tags
        self.observations = []  # List of vocabulary words
        self.start_prob = {}  # Initial probabilities \pi
        self.trans_prob = {}  # State transition probabilities A
        self.emit_prob = {}  # Emission probabilities B

    def train(self, annotated_corpus: List[Tuple[List[str], List[str]]]):
        """
        Train the HMM model from an annotated corpus.

        Parameters:
        annotated_corpus: List of tuples, where each tuple contains a list of words and corresponding POS tags.
        """
        start_counts = {}
        trans_counts = {}
        emit_counts = {}

        for sentence, tags in annotated_corpus:
            prev_tag = None
            for i, (word, tag) in enumerate(zip(sentence, tags)):
                # Update start probabilities
                if i == 0:
                    start_counts[tag] = start_counts.get(tag, 0) + 1

                # Update transition probabilities
                if prev_tag is not None:
                    trans_counts[(prev_tag, tag)] = trans_counts.get((prev_tag, tag), 0) + 1

                # Update emission probabilities
                emit_counts[(tag, word)] = emit_counts.get((tag, word), 0) + 1

                prev_tag = tag

        # Normalize start probabilities
        total_starts = sum(start_counts.values())
        self.start_prob = {tag: count / total_starts for tag, count in start_counts.items()}

        # Normalize transition probabilities
        tag_totals = {}
        for (prev_tag, tag), count in trans_counts.items():
            tag_totals[prev_tag] = tag_totals.get(prev_tag, 0) + count
        self.trans_prob = {key: count / tag_totals[key[0]] for key, count in trans_counts.items()}

        # Normalize emission probabilities
        tag_totals = {}
        for (tag, word), count in emit_counts.items():
            tag_totals[tag] = tag_totals.get(tag, 0) + count
        self.emit_prob = {key: count / tag_totals[key[0]] for key, count in emit_counts.items()}

        # Collect states and observations
        self.states = list(start_counts.keys())
        self.observations = list(set(word for _, word in emit_counts.keys()))

    def viterbi(self, sentence: List[str]) -> List[str]:
        """
        Perform POS tagging using the Viterbi algorithm.

        Parameters:
        sentence: List of words in the input sentence.

        Returns:
        List of POS tags corresponding to the input sentence.
        """
        V = [{}]
        path = {}

        # Initialization
        for state in self.states:
            V[0][state] = self.start_prob.get(state, 0) * self.emit_prob.get((state, sentence[0]), 0)
            path[state] = [state]

        # Recursion
        for t in range(1, len(sentence)):
            V.append({})
            new_path = {}

            for state in self.states:
                max_prob, best_prev_state = max(
                    (
                        V[t - 1][prev_state] * self.trans_prob.get((prev_state, state), 0) * self.emit_prob.get(
                            (state, sentence[t]), 0),
                        prev_state
                    )
                    for prev_state in self.states
                )

                V[t][state] = max_prob
                new_path[state] = path[best_prev_state] + [state]

            path = new_path

        # Termination
        max_prob, best_final_state = max((V[-1][state], state) for state in self.states)
        return path[best_final_state]

    def predict(self, sentences: List[List[str]]) -> List[List[str]]:
        """
        Predict POS tags for multiple sentences.

        Parameters:
        sentences: List of sentences, where each sentence is a list of words.

        Returns:
        List of predicted POS tags for each sentence.
        """
        return [self.viterbi(sentence) for sentence in sentences]


def preprocessing():
    sentences, pos_tags = extract_data_from_txt(input_file_path)
    print(" size of sentences: ", len(sentences))
    print(" size of pos_tags: ", len(pos_tags))
    annotated_corpus = [
        (sentence.split(), tag_list)
        for sentence, tag_list in zip(sentences, pos_tags)
    ]
    return annotated_corpus

def postprocess_output(sentence, predicted_tags):
    """
    Combine the sentence and the predicted tags for visualization.
    """
    words = sentence.split()
    return list(zip(words, predicted_tags[:len(words)]))

def test_model(sentence):
    annotated_corpus = preprocessing()
    tagger = HMMPOSTagger()
    tagger.train(annotated_corpus)
    tags = tagger.viterbi(sentence.split())
    return postprocess_output(sentence, tags)


# Example usage
if __name__ == "__main__":
    annotated_corpus = preprocessing()

    tagger = HMMPOSTagger()
    tagger.train(annotated_corpus)
    data = None
    with open(path.join(RAW_DATA_TAGS_ARABIC_PATH, RAW_DATA_TAGS_JSON), 'r', encoding='utf-8') as file:
        data = json.load(file)
    # Another example test sentence in Arabic with more than 10 words
    sentences = [
        "الطقس جميل اليوم",
        "الطقس جميل اليوم والجو حار",
        "الطقس جميل اليوم والجو حار جدا",
        "اسمي أحمد، وأنت",
        "أنا بخير، شكرا",
        "هل تحتاج إلى مساعدة",
        "أين تعيش",
        "أنا أعيش في القاهرة",
        "ما هو عملك",
        "أنا طالب",
        "ما هي هواياتك",
        "أحب القراءة والسفر",
        "يومك سعيد",
        "كيف يمكنني مساعدتك",
    ]
    for sent in sentences:
        test = sent.split()
        tags = tagger.viterbi(test)
        print("---------------")
        for idx in range(len(test)):
            print(f"{test[idx]}: {tags[idx]} :  {data[tags[idx]]}")
