"""Dataset class for the graph classification task."""

import os
from warnings import warn
from typing import Any

import torch
from torch.utils import data
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class GraphDataset(data.Dataset):
    """input and label image dataset"""

    def __init__(self,
                 root: str,
                 ids: list[str],
                 site: str | None = 'LUAD',
                 classdict: dict[str, int] | None = None,
                 target_patch_size: int | None = None,
                 use_refined_adj: bool =  False
                 ) -> None:
        """Create a GraphDataset.

        Args:
            root (str): Path to the dataset's root directory.
            ids (list[str]): List of ids of the images to load.
                Each id should be a string in the format "filename\tlabel".
            site (str | None): Name of the canonical tissue site the images from. The only sites
                that are recognized as canonical (i.e., they have a pre-defined classdict) are
                'LUAD', 'LSCC', 'NLST', and 'TCGA'. If your dataset is not a canonical site, leave
                this as None. 
            classdict (dict[str, int]): Dictionary mapping the class names to the class indices. Not
                needed if your dataset is a canonical site or your labels are already 0-indexed
                positive consecutive integers.
            target_patch_size (int | None): Size of the patches to extract from the images. (Not
                used.)
            use_refined_adj (bool): Whether to use refined adjacency matrices if available.
        """
        super(GraphDataset, self).__init__()
        self.root = root
        self.ids = ids
        self.use_refined_adj = use_refined_adj

        if classdict is not None:
            self.classdict = classdict
        else:
            if site is None:
                warn('Neither site nor classdict provided. Assuming class labels are integers.')
                self.classdict = None
            elif site in {'LUAD', 'LSCC'}:
                self.classdict = {'normal': 0, 'luad': 1, 'lscc': 2}
            elif site == 'NLST':
                self.classdict = {'normal': 0, 'tumor': 1}
            elif site == 'TCGA':
                self.classdict = {'Normal': 0, 'TCGA-LUAD': 1, 'TCGA-LUSC': 2}
            else:
                raise ValueError(f'Site {site} not recognized and classdict not provided')
        self.site = site

    def __getitem__(self, index: int) -> dict[str, Any]:
        info = self.ids[index].replace('\n', '')
        try:
            # Split by tab or multiple spaces to separate ID from label
            parts = info.split('\t') if '\t' in info else info.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid id format: {info}. Expected format is 'filename\tlabel'")
                
            graph_name = parts[0].strip()
            label = parts[1].strip().lower()
            
            # Simply use the root path directly - it should already point to the simclr_files directory
            graph_path = self.root.strip()
            
            sample = {}
            sample['label'] = self.classdict[label] if (self.classdict is not None) else int(label)
            sample['id'] = graph_name

            # Construct paths for features and adjacency
            feature_path = os.path.join(graph_path, graph_name, 'features.pt')
            if not os.path.exists(feature_path):
                # Try without trailing space if present
                alt_feature_path = os.path.join(graph_path.rstrip(), graph_name, 'features.pt')
                if os.path.exists(alt_feature_path):
                    feature_path = alt_feature_path
                else:
                    raise FileNotFoundError(f'features.pt for {graph_name} doesn\'t exist at {feature_path}')
            
            features = torch.load(feature_path, map_location='cpu')

            # Try to load refined adjacency first if use_refined_adj is True
            if self.use_refined_adj:
                refined_adj_path = os.path.join(graph_path, graph_name, 'refined_adj.pt')
                if os.path.exists(refined_adj_path):
                    adj_s = torch.load(refined_adj_path, map_location='cpu')
                else:
                    # Fall back to original adjacency
                    adj_s_path = os.path.join(graph_path, graph_name, 'adj_s.pt')
                    if not os.path.exists(adj_s_path):
                        # Try without trailing space if present
                        alt_adj_path = os.path.join(graph_path.rstrip(), graph_name, 'adj_s.pt')
                        if os.path.exists(alt_adj_path):
                            adj_s_path = alt_adj_path
                        else:
                            raise FileNotFoundError(f'adj_s.pt for {graph_name} doesn\'t exist at {adj_s_path}')
                    
                    adj_s = torch.load(adj_s_path, map_location='cpu')
            else:
                # Use original adjacency only
                adj_s_path = os.path.join(graph_path, graph_name, 'adj_s.pt')
                if not os.path.exists(adj_s_path):
                    # Try without trailing space if present
                    alt_adj_path = os.path.join(graph_path.rstrip(), graph_name, 'adj_s.pt')
                    if os.path.exists(alt_adj_path):
                        adj_s_path = alt_adj_path
                    else:
                        raise FileNotFoundError(f'adj_s.pt for {graph_name} doesn\'t exist at {adj_s_path}')
                
                adj_s = torch.load(adj_s_path, map_location='cpu')
            
            # Ensure dense tensor
            if adj_s.is_sparse:
                adj_s = adj_s.to_dense()

            sample['image'] = features
            sample['adj_s'] = adj_s

            return sample
            
        except Exception as e:
            print(f"Error processing {info}: {str(e)}")
            raise

    def __len__(self):
        return len(self.ids)
