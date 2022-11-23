# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocessing script before training the distilled model.
"""
import argparse
import gzip
import json
import logging
import os
import pickle
import shelve
from collections import Counter


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Token Counts for smoothing the masking probabilities in MLM (cf XLM/word2vec)"
    )
    parser.add_argument(
        "--dataloader_path", type=str, help="path to data db file."
    )
    parser.add_argument(
        "--token_counts_dump", type=str, default="/home/femi/compression/my_diacomp_dailyDialog/src/distillation/out/counts.pkl", help="The dump file."
    )
    parser.add_argument("--vocab_size", default=50257, type=int)
    args = parser.parse_args()

    logger.info(f"Loading data from {args.dataloader_path}")

   
    db = shelve.open(f'{args.dataloader_path}/db', 'r')
    keys = list(db.keys())
    for key in keys:
        chunk = json.loads(gzip.decompress(db[key]).decode('utf-8'))
        trunc_chunk = []
        for feat in chunk:
            trunc_chunk.append(feat['input_ids'])
    
    logger.info("Counting occurences for MLM.")
    
    counter = Counter()
    for input_ids in trunc_chunk:
        counter.update(input_ids)
    counts = [0] * args.vocab_size
    for k, v in counter.items():
        try:
            counts[k] = v
        except:
            print(k, v)
    logger.info(f"Dump to {args.token_counts_dump}")
    handle = open(args.token_counts_dump, "wb") 
    pickle.dump(counts, handle, protocol=pickle.HIGHEST_PROTOCOL)
