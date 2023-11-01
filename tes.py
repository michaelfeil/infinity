from fastembed.embedding import Embedding
import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.DataFrame(Embedding.list_supported_models())