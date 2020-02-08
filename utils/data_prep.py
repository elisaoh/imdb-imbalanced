from argparse import Namespace
from collections import Counter
import json
import os
import re
import string

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

class Vocabulary(object):
    """Class to process text and extract vocabulary for mapping"""

class ReviewVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""

class ReviewDataset(Dataset):