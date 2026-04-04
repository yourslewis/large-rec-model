"""
Debug entry point for model training without distributed processing.
Use this script for debugging with pdb:

python debug_main.py --gin_config_file=configs/your_debug_config.gin
"""

import logging
import os
import sys
import gin
import torch
import fbgemm_gpu  # noqa: F401, E402
import absl
from absl import app, flags
from data.reco_dataset_v3 import get_reco_dataset_v3
from trainer.train import Trainer
from trainer.util import make_model
from huggingface_hub import login

# Set absl logging config
absl.logging._warn_preinit_stderr = False
absl.logging.set_verbosity('info')
logging.basicConfig(level=logging.INFO)

# Define flags
flags.DEFINE_string("gin_config_file", None, "Path to the gin config file")
flags.DEFINE_string("data_path", None, "Path to the data directory")
flags.DEFINE_string("ads_semantic_embd_path", None, "Path to the precomputed embeddings for ads domain")
flags.DEFINE_string("web_browsing_semantic_embd_path", None, "Path to the precomputed embeddings for web browsing domain")
flags.DEFINE_string("shopping_semantic_embd_path", None, "Path to the precomputed embeddings for shopping domain")
flags.DEFINE_string("output_path", None, "Path to write the artifacts")

FLAGS = flags.FLAGS

def main(argv):
    hf_token = os.environ.get("HF_TOKEN", "hf_secret")
    login(token=hf_token)
    gin_config_file = FLAGS.gin_config_file
    data_path = FLAGS.data_path
    output_path = FLAGS.output_path
    
    # Set up embeddings path dict
    precomputed_embeddings_domain_to_dir = {
        0: FLAGS.ads_semantic_embd_path,
        1: FLAGS.web_browsing_semantic_embd_path,
        2: FLAGS.shopping_semantic_embd_path,
    }
    
    # In debug mode, we set these environment variables manually
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    
    # Force debug mode
    gin.parse_config("Trainer.debug_mode = True")
    
    if gin_config_file is not None:
        logging.info(f"Loading gin config from {gin_config_file}")
        gin.parse_config_file(gin_config_file)

    dataset = get_reco_dataset_v3(
        chronological=True,
        rank=0,
        world_size=1,
        data_path=data_path,
        debug_mode=True

    )
    
    model = make_model(
        dataset=dataset,
        precomputed_embeddings_domain_to_dir=precomputed_embeddings_domain_to_dir,
        debug_mode = True # Force debug mode for single process
    )
    
    trainer = Trainer(
        local_rank=0,
        rank=0,
        world_size=1,
        dataset=dataset,
        model=model,
        output_path=output_path,
    )

    trainer.train()

if __name__ == "__main__":
    app.run(main)