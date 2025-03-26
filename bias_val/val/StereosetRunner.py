import numpy as np
import torch
import pdb
import json
import string

from tqdm import tqdm
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Label(object):
    def __init__(self, human_id, label):
        """Label, represents a label object for a particular sentence.

        Args:
            human_id (`str`): Provides a unique ID for the human that labeled the sentence.
            label (`enum`): Provides a label for the sentence. This must be one of
                [stereotype, anti-stereotype, unrelated, related].
        """
        assert label in ["stereotype", "anti-stereotype", "unrelated", "related"]
        self.human_id = human_id
        self.label = label

class Sentence(object):
    def __init__(self, ID, sentence, labels, gold_label):
        """A generic sentence type that represents a sentence.

        Args:
            ID (`str`): Provides a unique ID for the sentence with respect to the example.
            sentence (`str`): The textual sentence.
            labels (`list` of `Label` objects): A list of human labels for the sentence.
            gold_label (`enum`): The gold label associated with this sentence,
                calculated by the argmax of the labels. This must be one of
                [stereotype, anti-stereotype, unrelated, related].
        """
        assert type(ID) == str
        assert gold_label in ["stereotype", "anti-stereotype", "unrelated"]
        assert isinstance(labels, list)
        assert isinstance(labels[0], Label)

        self.ID = ID
        self.sentence = sentence
        self.gold_label = gold_label
        self.labels = labels
        self.template_word = None

    def __str__(self):
        return f"{self.gold_label.capitalize()} Sentence: {self.sentence}"

class Example(object):
    def __init__(self, ID, bias_type, target, context, sentences):
        """A generic example.

        Args:
            ID (`str`): Provides a unique ID for the example.
            bias_type (`str`): Provides a description of the type of bias that is
                represented. It must be one of [RACE, RELIGION, GENDER, PROFESSION].
            target (`str`): Provides the word that is being stereotyped.
            context (`str`): Provides the context sentence, if exists,  that
                sets up the stereotype.
            sentences (`list`): A list of sentences that relate to the target.
        """
        self.ID = ID
        self.bias_type = bias_type
        self.target = target
        self.context = context
        self.sentences = sentences

class IntrasentenceExample(Example):
    def __init__(self, ID, bias_type, target, context, sentences):
        """Implements the Example class for an intrasentence example.

        See Example's docstring for more information.
        """
        super(IntrasentenceExample, self).__init__(
            ID, bias_type, target, context, sentences
        )

class IntersentenceExample(Example):
    def __init__(self, ID, bias_type, target, context, sentences):
        """
        Implements the Example class for an intersentence example.

        See Example's docstring for more information.
        """
        super(IntersentenceExample, self).__init__(
            ID, bias_type, target, context, sentences)


class StereosetLoader(object):
    def __init__(self, location, json_obj=None):
        """Instantiates the StereoSet object.

        Args:
            location (`str`): Location of the StereoSet.json file.
        """

        if json_obj == None:
            with open(location, "r") as f:
                self.json = json.load(f)
        else:
            self.json = json_obj

        self.version = self.json["version"]
        self.intrasentence_examples = self.__create_intrasentence_examples__(
            self.json["data"]["intrasentence"]
        )
        self.intersentence_examples = self.__create_intersentence_examples__(
            self.json['data']['intersentence'])


    def __create_intrasentence_examples__(self, examples):
        created_examples = []
        for example in examples:
            sentences = []
            for sentence in example["sentences"]:
                labels = []
                for label in sentence["labels"]:
                    labels.append(Label(**label))
                sentence_obj = Sentence(
                    sentence["id"], sentence["sentence"], labels, sentence["gold_label"]
                )
                word_idx = None
                for idx, word in enumerate(example["context"].split(" ")):
                    if "BLANK" in word:
                        word_idx = idx
                if word_idx is None:
                    raise Exception("No blank word found.")
                template_word = sentence["sentence"].split(" ")[word_idx]
                sentence_obj.template_word = template_word.translate(
                    str.maketrans("", "", string.punctuation)
                )
                sentences.append(sentence_obj)
            created_example = IntrasentenceExample(
                example["id"],
                example["bias_type"],
                example["target"],
                example["context"],
                sentences,
            )
            created_examples.append(created_example)
        return created_examples
    
    def __create_intersentence_examples__(self, examples):
        created_examples = []
        for example in examples:
            sentences = []
            for sentence in example['sentences']:
                labels = []
                for label in sentence['labels']:
                    labels.append(Label(**label))
                sentence = Sentence(
                    sentence['id'], sentence['sentence'], labels, sentence['gold_label'])
                sentences.append(sentence)
            created_example = IntersentenceExample(
                example['id'], example['bias_type'], example['target'], 
                example['context'], sentences) 
            created_examples.append(created_example)
        return created_examples

    def get_intrasentence_examples(self):
        return self.intrasentence_examples

    def get_intersentence_examples(self):
        return self.intersentence_examples



class Runner:

    def __init__(
        self,
        model,
        tokenizer,
        model_name,
        input_file,
        batch_size=1,
        max_seq_length=128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.input_file = input_file
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

    def __call__(self):
        bias = {}
        print("开始评估Stereoset:")
        intrasentence_bias = self.evaluate_intrasentence()
        intersentence_bias = self.evaluate_intersentence()
        bias["intrasentence"] = intrasentence_bias
        bias["intersentence"] = intersentence_bias

        return bias
    
    def evaluate_intrasentence(self):
        # pdb.set_trace()
        model = self.model.to(device)
        stereoset = StereosetLoader(self.input_file)
        # unconditional_start_token = "<|endoftext|>"
        # IF opt, use this 
        unconditional_start_token = "</s>"
        start_token = (
            torch.tensor(self.tokenizer.encode(unconditional_start_token))[-1].unsqueeze(0)
            .to(device)
            .unsqueeze(0)
        )

        with torch.no_grad():
            initial_token_probabilities = model(start_token)
            
        initial_token_probabilities = torch.softmax(
            initial_token_probabilities[0], dim=-1
        )
        assert initial_token_probabilities.shape[0] == 1
        assert initial_token_probabilities.shape[1] == 1

        clusters = stereoset.get_intrasentence_examples()
        predictions = []
        for cluster in tqdm(clusters):
            joint_sentence_probability = []
            for sentence in cluster.sentences:
                probabilities = {}
                tokens = self.tokenizer.encode(sentence.sentence)
                tokens_tensor = torch.tensor(tokens).to(device).unsqueeze(0)

                with torch.no_grad():
                    joint_sentence_probability = [
                        initial_token_probabilities[0, 0, tokens[0]].item()
                    ]

                    output = torch.softmax(model(tokens_tensor)[0], dim=-1)

                for idx in range(1, len(tokens)):
                    joint_sentence_probability.append(
                        output[0, idx - 1, tokens[idx]].item()
                    )

                assert len(tokens) == len(joint_sentence_probability)

                score = np.sum([np.log2(i) for i in joint_sentence_probability])
                score /= len(joint_sentence_probability)
                score = np.power(2, score)

                probabilities["id"] = sentence.ID
                probabilities["score"] = score

                predictions.append(probabilities)

        return predictions
        
    def evaluate_intersentence(self):
        model = self.model.to(device)
        stereoset = StereosetLoader(self.input_file)
        # unconditional_start_token = "<|endoftext|>"
        unconditional_start_token = "<|endoftext|>"
        start_token = (
            torch.tensor(self.tokenizer.encode(unconditional_start_token))[-1].unsqueeze(0)
            .to(device)
            .unsqueeze(0)
        )

        with torch.no_grad():
            initial_token_probabilities = model(start_token)
            
        initial_token_probabilities = torch.softmax(
            initial_token_probabilities[0], dim=-1
        )
        assert initial_token_probabilities.shape[0] == 1
        assert initial_token_probabilities.shape[1] == 1

        clusters = stereoset.get_intersentence_examples()
        predictions = []
        for cluster in tqdm(clusters):
            joint_sentence_probability = []
            for sentence in cluster.sentences:
                probabilities = {}
                tokens = self.tokenizer.encode(sentence.sentence)
                tokens_tensor = torch.tensor(tokens).to(device).unsqueeze(0)

                with torch.no_grad():
                    joint_sentence_probability = [
                        initial_token_probabilities[0, 0, tokens[0]].item()
                    ]

                    output = torch.softmax(model(tokens_tensor)[0], dim=-1)

                for idx in range(1, len(tokens)):
                    joint_sentence_probability.append(
                        output[0, idx - 1, tokens[idx]].item()
                    )

                assert len(tokens) == len(joint_sentence_probability)

                score = np.sum([np.log2(i) for i in joint_sentence_probability])
                score /= len(joint_sentence_probability)
                score = np.power(2, score)

                probabilities["id"] = sentence.ID
                probabilities["score"] = score

                predictions.append(probabilities)

        return predictions
        