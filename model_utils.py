"""
model_utils.py
--------------
Shared utilities used by train.py and loaded when unpickling saved models.

app.py loads the model with:
    pickle.load(open("pickle/model.pkl", "rb"))

For this to work when the model was trained by train.py, the LabelDecodingWrapper
class must be importable at unpickling time.  Placing it here (not inside train.py)
ensures that ``import model_utils`` succeeds from any script in the project root.
"""


class LabelDecodingWrapper:
    """Wraps a classifier that was trained on encoded integer labels and maps
    predictions back to the original label space (e.g. {0, 1} -> {-1, 1}).

    This wrapper is picklable because it lives at module level in model_utils.py.
    """

    def __init__(self, estimator, label_encoder):
        # Use object.__setattr__ to avoid triggering __getattr__ during init
        object.__setattr__(self, '_est', estimator)
        object.__setattr__(self, '_le', label_encoder)

    def __getattr__(self, name):
        # Only called for attributes not found in instance/class __dict__
        return getattr(object.__getattribute__(self, '_est'), name)

    def predict(self, X):
        est = object.__getattribute__(self, '_est')
        le  = object.__getattribute__(self, '_le')
        encoded = est.predict(X)
        return le.inverse_transform(encoded)

    def predict_proba(self, X):
        est = object.__getattribute__(self, '_est')
        return est.predict_proba(X)
