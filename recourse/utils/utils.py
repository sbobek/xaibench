from typing import Dict, List
import numpy as np

def prepare_ds_lore(df, name = 'dataset', class_name='class', discrete=[], label_encoder={}):
    features = [f for f in df.columns if f not in [class_name]]
    
    dataset = {}
    dataset['name'] = name
    dataset['df'] = df
    dataset['columns'] = list(df.columns)
    dataset['class_name'] = class_name
    dataset['possible_outcomes'] = list(np.unique(df[class_name]))
    
    types = {}
    types['integer'] = [c for c,t in zip(df.columns, df.dtypes) if 'int' in str(t)]
    types['double'] = [c for c,t in zip(df.columns, df.dtypes) if 'float' in str(t)]
    types['string'] = [c for c,t in zip(df.columns, df.dtypes) if 'str' in str(t) or 'object' in str(t)]
    dataset['type_features'] = types
    
    
    typemap={}
    typemap['object'] = 'string'
    typemap['float64'] = 'double'
    typemap['float32'] = 'double'
    typemap['int32'] = 'integer'
    typemap['float'] = 'double'
    typemap['int'] = 'integer'
    typemap['int64'] = 'integer'
    dataset['features_type'] = dict(zip(df.columns, map(lambda x: typemap[str(x)],df.dtypes)))
    
    dataset['discrete'] = list(df[features].columns[discrete])
    dataset['continuous'] = [f for f in features if f not in dataset['discrete']]
    dataset['idx_features'] = dict(enumerate(df[features].columns))
    dataset['label_encoder'] = label_encoder
    dataset['discrete_indices'] = [list(df.columns).index(f) for f in dataset['discrete']] 
    dataset['discrete_names'] = dict(zip(dataset['discrete_indices'],[np.unique(df[features[i]]) for i in dataset['discrete_indices']]))
    dataset['feature_names'] = features
    dataset['X'] =  df[dataset['feature_names']].values
    dataset['y'] = df[dataset['class_name']].values
    
    return dataset

def merge_default_parameters(hyperparams: Dict, default: Dict) -> Dict:
    """
    Checks if the input parameter hyperparams contains every necessary key and if not, uses default values or
    raises a ValueError if no default value is given.

    Parameters
    ----------
    hyperparams: dict
        Hyperparameter as passed to the recuorse method.
    default: dict
        Dictionary with every necessary key and default value.
        If key has no default value and hyperparams has no value for it, raise a ValueError

    Returns
    -------
    dict
        Dictionary with every necessary key.
    """
    keys = default.keys()
    dict_output = dict()

    for key in keys:
        if isinstance(default[key], dict):
            hyperparams[key] = (
                dict() if key not in hyperparams.keys() else hyperparams[key]
            )
            sub_dict = merge_default_parameters(hyperparams[key], default[key])
            dict_output[key] = sub_dict
            continue
        if key not in hyperparams.keys():
            default_val = default[key]
            if default_val is None:
                # None value for key depicts that user has to pass this value in hyperparams
                raise ValueError(
                    "For {} is no default value defined, please pass this key and its value in hyperparams".format(
                        key
                    )
                )
            elif isinstance(default_val, str) and default_val == "_optional_":
                # _optional_ depicts that value for this key is optional and therefore None
                default_val = None
            dict_output[key] = default_val
        else:
            if hyperparams[key] is None:
                raise ValueError("For {} in hyperparams is a value needed".format(key))
            dict_output[key] = hyperparams[key]

    return dict_output
