import csv
import warnings
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pdb

from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Runner:
    def __init__(self, model, tokenizer, input_file, bias_type=None):
        self.model = model
        self.tokenizer = tokenizer
        self.input_file = input_file
        self.bias_type = bias_type if bias_type != "race" else "race-color"

    def _read_data(self, input_file):
        
        # pdb.set_trace()
        df_data = pd.DataFrame(columns=["sent1", "sent2", "direction", "bias_type"])

        if self.bias_type is not None:
            print(f"Evaluating {self._bias_type} examples.")

        with open(input_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                direction, gold_bias = "_", "_"
                direction = row["stereo_antistereo"]
                bias_type = row["bias_type"]

                if self.bias_type is not None and bias_type != self.bias_type:
                    continue

                sent1, sent2 = "", ""
                if direction == "stereo":
                    sent1 = row["sent_more"]
                    sent2 = row["sent_less"]
                else:
                    sent1 = row["sent_less"]
                    sent2 = row["sent_more"]

                # pdb.set_trace()
                df_item = {
                    "sent1": sent1,
                    "sent2": sent2,
                    "direction": direction,
                    "bias_type": bias_type,
                }
                # df_data = df_data.append(df_item, ignore_index=True)
                # pdb.set_trace()
                df_item = pd.DataFrame([df_item])
                df_data = pd.concat([df_data, df_item], ignore_index=True)

        return df_data
    
    def __call__(self):
        df_data = self._read_data(self.input_file)

        self.model.to(device)

        # df_score = pd.DataFrame(
        #     columns=[
        #         "sent_more",
        #         "sent_less",
        #         "sent_more_score",
        #         "sent_less_score",
        #         "score",
        #         "stereo_antistereo",
        #         "bias_type",
        #     ]
        # )

        total_stereo, total_antistereo = 0, 0
        stereo_score, antistereo_score = 0, 0

        N = 0
        neutral = 0
        total = len(df_data.index)

        with tqdm(total=total) as pbar:
            for index, data in df_data.iterrows():
                direction = data["direction"]
                bias = data["bias_type"]

                sent1, sent2 = data["sent1"], data["sent2"]

                sent1_token_ids = self.tokenizer.encode(sent1)
                sent2_token_ids = self.tokenizer.encode(sent2)

                score1 = self._joint_log_probability(sent1_token_ids)
                score2 = self._joint_log_probability(sent2_token_ids)

                N += 1
                pair_score = 0
                pbar.update(1)
                if score1 == score2:
                    neutral += 1
                else:
                    if direction == "stereo":
                        total_stereo += 1
                        if score1 > score2:
                            stereo_score += 1
                            pair_score = 1
                    elif direction == "antistereo":
                        total_antistereo += 1
                        if score2 > score1:
                            antistereo_score += 1
                            pair_score = 1

                sent_more, sent_less = "", ""
                if direction == "stereo":
                    sent_more = data["sent1"]
                    sent_less = data["sent2"]
                    sent_more_score = score1
                    sent_less_score = score2
                else:
                    sent_more = data["sent2"]
                    sent_less = data["sent1"]
                    sent_more_score = score2
                    sent_less_score = score1

                df_itm = {
                    "sent_more": sent_more,
                    "sent_less": sent_less,
                    "sent_more_score": sent_more_score,
                    "sent_less_score": sent_less_score,
                    "score": pair_score,
                    "stereo_antistereo": direction,
                    "bias_type": bias,
                }

                df_itm = pd.DataFrame([df_itm])
                
                df_data = pd.concat([df_data, df_itm], ignore_index=True)

        print("=" * 100)
        print("Total examples:", N)
        print("Metric score:", round((stereo_score + antistereo_score) / N * 100, 2))
        print("Stereotype score:", round(stereo_score / total_stereo * 100, 2))
        if antistereo_score != 0:
            print(
                "Anti-stereotype score:",
                round(antistereo_score / total_antistereo * 100, 2),
            )
        print("Num. neutral:", round(neutral / N * 100, 2))
        print("=" * 100)
        print()

        return round((stereo_score + antistereo_score) / N * 100, 2)
   
    def _joint_log_probability(self, tokens):
        start_token = (
            torch.tensor(self.tokenizer.encode("<|endoftext|>"))
            .to(device)
            .unsqueeze(0)
        )
        initial_token_probabilities = self.model(start_token)
        initial_token_probabilities = torch.softmax(
            initial_token_probabilities[0], dim=-1
        )

        tokens_tensor = torch.tensor(tokens).to(device).unsqueeze(0)

        with torch.no_grad():
                joint_sentence_probability = [
                    initial_token_probabilities[0, 0, tokens[0]].item()
                ]

                output = torch.softmax(self.model(tokens_tensor)[0], dim=-1)
        
        for idx in range(1, len(tokens)):
            joint_sentence_probability.append(
                output[0, idx - 1, tokens[idx]].item()
            )
                
        assert len(tokens) == len(joint_sentence_probability)

        score = np.sum([np.log2(i) for i in joint_sentence_probability])
        score /= len(joint_sentence_probability)
        score = np.power(2, score)

        return score