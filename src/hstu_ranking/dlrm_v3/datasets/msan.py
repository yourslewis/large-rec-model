# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe
import json
from typing import Any, Dict, List, Iterator, Optional

import pandas as pd
import torch
from dlrm_v3.datasets.utils import (
    maybe_truncate_seq,
    separate_uih_candidates,
)
from modules.dlrm_hstu import DlrmHSTUConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torch.utils.data import IterableDataset


def process_and_hash_x(x: Any, hash_size: int) -> Any:
    if isinstance(x, str):
        x = json.loads(x)
    if isinstance(x, list):
        return [x_i % hash_size for x_i in x]
    else:
        return x % hash_size



class DLRMv3MSANIterableDataset(IterableDataset):
    """
    Iterable version of MSAN dataset that supports large-scale streaming data processing
    
    Maintains the same output format as the original DLRMv3MSANDataset
    """
    
    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        # embedding_config: Dict[str, Any],
        seq_logs_file: str,
        is_inference: bool,
        max_lines: Optional[int] = None,
        skip_lines: int = 0,
        rank: int = 0,
        world_size: int = 1,
        mount_path: str = None,  # Add mount_path parameter
        **kwargs,
    ) -> None:
        """
        Initialize iterable dataset
        
        Args:
            hstu_config: HSTU configuration
            embedding_config: Embedding table configuration
            seq_logs_file: Data file path
            is_inference: Whether in inference mode
            max_lines: Maximum number of lines to read (for testing)
            skip_lines: Number of lines to skip (for train/test split)
            rank: Rank in distributed training
            world_size: World size in distributed training
        """
        # Copy necessary attributes from DLRMv3RandomDataset
        self._hstu_config = hstu_config
        self._is_inference = is_inference
        self._max_num_candidates = hstu_config.max_num_candidates
        self._max_num_candidates_inference = hstu_config.max_num_candidates_inference
        
        # Calculate max_uih_len, keeping consistent with base class DLRMv3RandomDataset
        self._max_seq_len = hstu_config.max_seq_len
        self._contextual_feature_to_max_length = hstu_config.contextual_feature_to_max_length or {}
        self._max_uih_len = (
            self._max_seq_len
            - self._max_num_candidates
            - len(self._contextual_feature_to_max_length)
            # if self._contextual_feature_to_max_length
            # else 0
        )
        
        # Set feature keys
        self._uih_keys = hstu_config.hstu_uih_feature_names
        self._candidates_keys = hstu_config.hstu_candidate_feature_names
        # self._contextual_feature_to_max_length = hstu_config.contextual_feature_to_max_length or {}
        
        # self.embedding_config = embedding_config
        self.seq_logs_file = seq_logs_file
        self.max_lines = max_lines
        self.skip_lines = skip_lines
        self.rank = rank
        self.world_size = world_size
        
        # Prepare local data path
        import fsspec
        import os
        
        self.fs = fsspec.filesystem('file')
        
        # mount_path uses hardcoded default value
        if not mount_path:
            mount_path = "/scratch/azureml/cr/j/662dba604077490d886ddb94c5cf03ad/cap/data-capability/wd/INPUT_data"
        
        self.mount = mount_path
        self.data_path = os.path.join(self.mount, seq_logs_file)
    
    def load_item(self, data, max_num_candidates):
        """
        Same load_item method as the original class
        """
        with torch.profiler.record_function("load_item"):
            video_history_uih, video_history_candidates = separate_uih_candidates(
                data.event_id,
                candidates_max_seq_len=max_num_candidates,
            )
            action_weights_uih, action_weights_candidates = separate_uih_candidates(
                data.action_weights,
                candidates_max_seq_len=max_num_candidates,
            )
            timestamps_uih, _ = separate_uih_candidates(
                # data.timestamps_unix,
                data.timestamps_constant,
                candidates_max_seq_len=max_num_candidates,
            )
            
            # self._max_uih_len = 2056-10-6 = 2040
            video_history_uih = maybe_truncate_seq(video_history_uih, self._max_uih_len) 
            timestamps_uih = maybe_truncate_seq(timestamps_uih, self._max_uih_len)

            uih_seq_len = len(video_history_uih)
            assert uih_seq_len == len(
                timestamps_uih
            ), "history len differs from timestamp len."

            uih_kjt_values: List[torch.Tensor] = []
            uih_kjt_lengths: List[torch.Tensor] = []
            for name, length in self._contextual_feature_to_max_length.items():
                uih_kjt_values.append(data[name])
                uih_kjt_lengths.append(length)

            pseudo_action_weights_uih = [0.0 for _ in range(uih_seq_len)]
            pseudo_watch_times_uih = [0.0 for _ in range(uih_seq_len)]

            uih_kjt_values.extend(
                video_history_uih 
                + timestamps_uih 
                + pseudo_action_weights_uih 
                + pseudo_watch_times_uih
            )

            uih_kjt_lengths.extend(
                [
                    uih_seq_len
                    for _ in range(
                        len(self._uih_keys) # self._uih_keys = hstu_config.hstu_uih_feature_names
                        - len(self._contextual_feature_to_max_length)
                    )
                ]
            )

            dummy_query_time = max(timestamps_uih)
            uih_features_kjt = KeyedJaggedTensor(
                keys=self._uih_keys,
                lengths=torch.tensor(uih_kjt_lengths).long(),
                values=torch.tensor(uih_kjt_values).long(),
            )

            candidates_kjt_lengths = max_num_candidates * torch.ones(
                len(self._candidates_keys)
            )
            candidates_kjt_values = (
                video_history_candidates
                + action_weights_candidates
                + [1] * max_num_candidates  # item_pseudo_watchtime
                + [dummy_query_time] * max_num_candidates
            )
            candidates_features_kjt = KeyedJaggedTensor(
                keys=self._candidates_keys,
                lengths=candidates_kjt_lengths.clone().detach().long(),
                values=torch.tensor(candidates_kjt_values).long(),
            )

        return uih_features_kjt, candidates_features_kjt
    
    def __iter__(self) -> Iterator[tuple]:
        """
        Iterable implementation: read and process data line by line
        """
        from torch.utils.data import get_worker_info
        
        # Get worker info for distributed processing
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        

        # Calculate global worker ID
        global_worker_id = self.rank * num_workers + worker_id
        global_num_workers = self.world_size * num_workers
        
        max_num_candidates = (
            self._max_num_candidates_inference
            if self._is_inference
            else self._max_num_candidates
        )
        
        sample_idx = 0
        processed_count = 0
        
        try:
            # Open data stream
            import pandas as pd
            
            # Use chunksize to read data in chunks
            chunk_size = 10000  # Read 10000 lines at a time

            # Calculate actual rows to skip and read
            skiprows = self.skip_lines if self.skip_lines > 0 else None
            nrows = self.max_lines
            
            with self.fs.open(self.data_path) as f:
                for chunk in pd.read_csv(
                    f, 
                    sep='\t',
                    names=['sequence_id', 'task_id', 'label', 'events', 'demand_type'],
                    chunksize=chunk_size,
                    skiprows=skiprows,
                    nrows=nrows
                ):
                    # Process each row in the current chunk
                    for _, row_data in chunk.iterrows():
                        # Distributed data sharding: only process samples assigned to current worker
                        if sample_idx % global_num_workers == global_worker_id:
                            try:
                                # Process data row
                                processed_row = self._process_raw_row(row_data)
                                if processed_row is not None:
                                    # Check if sequence length meets requirements
                                    if len(processed_row.event_id) > max_num_candidates:
                                        # Use the same load_item method as the original class
                                        sample = self.load_item(processed_row, max_num_candidates)
                                        yield sample
                                        processed_count += 1
                            except Exception as e:
                                print(f"Error processing sample {sample_idx}: {e}")
                                continue
                        
                        sample_idx += 1
                        
                        # Exit if maximum lines limit is reached
                        if self.max_lines and sample_idx >= self.max_lines:
                            break
                    
                    # Exit outer loop if maximum lines limit is reached
                    if self.max_lines and sample_idx >= self.max_lines:
                        break
        
        except Exception as e:
            print(f"Error reading data file: {e}")
            return
        
        if global_worker_id == 0:
            print(f"Worker {global_worker_id} processed {processed_count} samples from {sample_idx} total samples")
    
    def _process_raw_row(self, row_data: pd.Series) -> Optional[pd.Series]:
        """
        Process raw data row and convert to required format
        """
        try:
            import numpy as np
            from datetime import datetime
            
            # Handle missing values
            if row_data.isna().any():
                return None
            
            # Convert data types
            sequence_id = row_data['sequence_id']
            task_id = row_data['task_id']
            label = int(row_data['label'])
            events = row_data['events']
            demand_type = int(row_data['demand_type'])
            
            # Process events - convert to integer list and reverse
            event_ids = [int(i) for i in events.split()][::-1]
            
            timestamps_constant = [1] * len(event_ids)
            
            # Generate action weights
            seq_length = len(event_ids) - 1
            action_weights = [0] * seq_length + [label]
            
            # Create data row
            row_dict = {
                'seq_id': sequence_id,
                'event_id': event_ids,
                'action_weights': action_weights,
                # 'timestamps_unix': timestamps_unix,
                'timestamps_constant': timestamps_constant,
                'task_id': task_id,
                'demand_type': demand_type
            }
            
            
            return pd.Series(row_dict)
            
        except Exception as e:
            print(f"Error processing row: {e}")
            return None


