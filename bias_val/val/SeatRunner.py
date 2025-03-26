import json
import os
import random
import re
import torch
import pdb

import numpy as np
from val.weat import run_test

class Runner:
    TEST_EXT = ".jsonl"
    def __init__(self, model, tokenizer, tests, data_dir, n_sample=100000, parametric=False, seed=0):
        self.model = model
        self.tokenizer = tokenizer
        self.tests = tests
        self.data_dir = data_dir
        self.n_sample = n_sample
        self.parametric = parametric
        self.seed = seed

    def __call__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        # pdb.set_trace()

        all_tests = sorted(
            [
                entry[: -len(self.TEST_EXT)]
                for entry in os.listdir(self.data_dir)
                if not entry.startswith(".") and entry.endswith(self.TEST_EXT)
            ],
            key=_test_sort_key,
        )

        tests = self.tests or all_tests

        results = []
        for test in tests:
            print(f"Running test {test}")

            # pdb.set_trace()
            encs = _load_json(os.path.join(self.data_dir, f"{test}{self.TEST_EXT}"))

            print("Computing sentence encodings")
            encs_targ1 = _encode(
                self.model, self.tokenizer, encs["targ1"]["examples"]
            )
            encs_targ2 = _encode(
                self.model, self.tokenizer, encs["targ2"]["examples"]
            )
            encs_attr1 = _encode(
                self.model, self.tokenizer, encs["attr1"]["examples"]
            )
            encs_attr2 = _encode(
                self.model, self.tokenizer, encs["attr2"]["examples"]
            )

            encs["targ1"]["encs"] = encs_targ1
            encs["targ2"]["encs"] = encs_targ2
            encs["attr1"]["encs"] = encs_attr1
            encs["attr2"]["encs"] = encs_attr2

            print("\tDone!")

            # Run the test on the encodings.
            esize, pval = run_test(
                encs, n_samples=self.n_sample, parametric=self.parametric
            )

            results.append(
                {
                    "test": test,
                    "p_value": pval,
                    "effect_size": esize,
                }
            )

        return results

def _test_sort_key(test):
    """Return tuple to be used as a sort key for the specified test name.
    Break test name into pieces consisting of the integers in the name
    and the strings in between them.
    """
    key = ()
    prev_end = 0
    for match in re.finditer(r"\d+", test):
        key = key + (test[prev_end : match.start()], int(match.group(0)))
        prev_end = match.end()
    key = key + (test[prev_end:],)

    return key


def _load_json(sent_file):
    """Load from json. We expect a certain format later, so do some post processing."""
    print(f"Loading {sent_file}...")
    all_data = json.load(open(sent_file, "r"))
    data = {}
    for k, v in all_data.items():
        examples = v["examples"]
        data[k] = examples
        v["examples"] = examples

    return all_data

def _encode(model, tokenizer, texts):
    encs = {}
    for text in texts:
        # Encode each example.
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)

        # Average over the last layer of hidden representations.
        enc = outputs["last_hidden_state"]
        enc = enc.mean(dim=1)

        # Following May et al., normalize the representation.
        encs[text] = enc.detach().view(-1).numpy()
        encs[text] /= np.linalg.norm(encs[text])

    return encs