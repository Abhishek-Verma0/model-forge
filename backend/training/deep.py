"""Deep-learning models for tables (plan §9.4), in PyTorch, as scikit-learn estimators.

The MLP (scikit-learn) and the autoencoder (preprocessing/advanced.py) already exist;
these five complete the plan's list. Each speaks the MLP's interface -- partial_fit =
one epoch, loss_, max_iter = epochs, tol -- so the MLP's epoch loop (training/epochs.py)
drives them: live loss and accuracy curves, validation on held-back rows, early
stopping. A plain fit() trains max_iter epochs (ensembles, learning curves).

How a table row goes in: every column becomes a learned vector, value x weight +
bias (the "numerical feature tokenizer" of FT-Transformer, Gorishniy et al. 2021),
so a network sees "column 1, column 2, ..." as a sequence of tokens.

  lstm / gru       a recurrent network reads the column tokens one after another and
                   predicts from its final memory (LSTM: 3 gates; GRU: 2, lighter).
  cnn1d            1-D convolutions slide over neighbouring column tokens, then average.
  tabnet           TabNet (Arik & Pfister, AAAI 2021): in each decision step a sparse
                   attention mask (sparsemax) picks the columns that step looks at;
                   steps share part of their feature transformer; outputs are summed.
                   Written here (the pytorch-tabnet package is not installed), so it is
                   called TabNet-style; defaults are pytorch-tabnet's published ones.
  tab_transformer  every column token (+ a [CLS] token) goes through a pre-norm
                   Transformer encoder; the head reads [CLS] -- FT-Transformer, the
                   form of the TabTransformer family that takes numeric columns (the
                   original TabTransformer, Huang et al. 2020, attends over category
                   columns only, and after preprocessing every column is a number).

Recurrent and convolutional models treat column ORDER as meaningful, which in a
table it is not; on ordinary tables gradient-boosted trees usually win (Grinsztajn
et al., NeurIPS 2022) -- the comparison table shows it on the user's data.

Inputs are standardised inside the model (training rows' mean / sd), so it works
whether or not the Preprocessing step scaled them. The GPU is used whenever PyTorch
sees one (rule: no CPU fallback); a saved model holds its weights on the CPU, so it
loads on any machine and moves to the GPU when it predicts there.
"""

import copy
import math

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from torch import nn
from torch.nn import functional as F

HEADS = 4                  # our starting point: attention heads (fewer when the hidden size isn't a multiple)
TABNET_GAMMA = 1.3         # published default: pytorch-tabnet gamma (mask relaxation)
TABNET_SPARSITY = 1e-3     # published default: pytorch-tabnet lambda_sparse
MAX_TOKENS = 512           # our starting point: most input columns a token model takes (memory: rows x columns x size)
PREDICT_BATCH = 4096       # our starting point: rows per forward pass when predicting
# attention keeps rows x heads x (columns + 1)^2 scores at once: measured 2026-11-01 (review, CPU, 512 columns)
# ~1.1 GB for 256 rows, so a 4,096-row batch would need ~17 GB. The transformer's predict batch is sized from this.
ATTENTION_CELLS = 50_000_000   # our starting point: attention scores per forward pass (~200 MB as float32)
EPS = 1e-15


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tokens(nn.Module):
    """Column j's value -> x_j * W_j + b_j (FT-Transformer's numerical embedding and init)."""

    def __init__(self, n, dim):
        super().__init__()
        bound = dim ** -0.5
        self.weight = nn.Parameter(torch.empty(n, dim).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(n, dim).uniform_(-bound, bound))

    def forward(self, x):
        return x[:, :, None] * self.weight + self.bias            # rows x columns x dim


class Recurrent(nn.Module):
    """Stacked one-layer LSTM / GRU cells with dropout between them -- what num_layers +
    dropout does, but without cuDNN's own RNN dropout state: measured 2026-11-01 (torch
    2.11 + CUDA 12.8, Windows), a process that trained a cuDNN RNN with inter-layer
    dropout crashed when it exited (0xC0000409); 1 layer, or no dropout, exited cleanly."""

    def __init__(self, n, out, hidden, layers, dropout, cell):
        super().__init__()
        self.tokens = Tokens(n, hidden)
        cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnns = nn.ModuleList([cls(hidden, hidden, batch_first=True) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, out)

    def forward(self, x):
        seq = self.tokens(x)
        for rnn in self.rnns:
            seq = self.drop(rnn(seq)[0])          # dropout after every layer, the last one before the head
        return self.head(seq[:, -1]), 0.0


class Conv(nn.Module):
    def __init__(self, n, out, hidden, layers, dropout):
        super().__init__()
        self.tokens = Tokens(n, hidden)
        blocks = []
        for _ in range(layers):
            blocks += [nn.Conv1d(hidden, hidden, 3, padding=1), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)]
        self.conv = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, out)

    def forward(self, x):
        h = self.conv(self.tokens(x).transpose(1, 2))           # rows x size x columns
        return self.head(h.mean(dim=2)), 0.0


class Transformer(nn.Module):
    def __init__(self, n, out, hidden, layers, dropout):
        super().__init__()
        heads = next(h for h in (HEADS, 2, 1) if hidden % h == 0)
        self.tokens = Tokens(n, hidden)
        self.cls = nn.Parameter(torch.empty(1, 1, hidden).uniform_(-hidden ** -0.5, hidden ** -0.5))
        layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=2 * hidden, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, out))

    def forward(self, x):
        t = self.tokens(x)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], dim=1)
        return self.head(self.encoder(t)[:, 0]), 0.0


def sparsemax(z):
    """Martins & Astudillo (2016): the Euclidean projection of each row onto the simplex --
    like softmax, but most entries come out exactly 0."""
    zs = torch.sort(z, dim=1, descending=True)[0]
    k = torch.arange(1, z.shape[1] + 1, device=z.device, dtype=z.dtype)
    cs = zs.cumsum(dim=1)
    kz = ((1 + k * zs) > cs).sum(dim=1, keepdim=True)          # support size
    tau = (cs.gather(1, kz - 1) - 1) / kz.to(z.dtype)
    return torch.clamp(z - tau, min=0)


class GLU(nn.Module):
    def __init__(self, i, o, fc=None):
        super().__init__()
        self.fc, self.bn, self.o = fc or nn.Linear(i, 2 * o, bias=False), nn.BatchNorm1d(2 * o), o

    def forward(self, x):
        h = self.bn(self.fc(x))
        return h[:, :self.o] * torch.sigmoid(h[:, self.o:])


class FeatureTransformer(nn.Module):
    """TabNet's feature transformer: 2 GLU layers shared by every step + 2 of its own,
    residual connections scaled by sqrt(0.5)."""

    def __init__(self, shared, i, o):
        super().__init__()
        self.blocks = nn.ModuleList([GLU(i, o, shared[0]), GLU(o, o, shared[1]), GLU(o, o), GLU(o, o)])

    def forward(self, x):
        h = self.blocks[0](x)
        for b in self.blocks[1:]:
            h = (h + b(h)) * math.sqrt(0.5)
        return h


class TabNet(nn.Module):
    def __init__(self, n, out, hidden, layers, dropout):
        super().__init__()
        nd = na = hidden                                        # decision / attention widths (n_d = n_a)
        width = nd + na
        self.bn = nn.BatchNorm1d(n)
        shared = [nn.Linear(n, 2 * width, bias=False), nn.Linear(width, 2 * width, bias=False)]
        self.first = FeatureTransformer(shared, n, width)
        self.steps = nn.ModuleList([FeatureTransformer(shared, n, width) for _ in range(layers)])
        self.attention = nn.ModuleList([nn.Sequential(nn.Linear(na, n, bias=False), nn.BatchNorm1d(n))
                                        for _ in range(layers)])
        self.drop, self.head, self.nd = nn.Dropout(dropout), nn.Linear(nd, out), nd

    def forward(self, x):
        x = self.bn(x)
        prior = torch.ones_like(x)
        a = self.first(x)[:, self.nd:]
        out, entropy = 0, 0
        for step, att in zip(self.steps, self.attention):
            mask = sparsemax(att(a) * prior)
            prior = prior * (TABNET_GAMMA - mask)
            entropy = entropy + (-mask * torch.log(mask + EPS)).sum(dim=1).mean()
            h = step(mask * x)
            out, a = out + torch.relu(h[:, :self.nd]), h[:, self.nd:]
        return self.head(self.drop(out)), entropy / len(self.steps)     # sparsity penalty: mask entropy


ARCHS = ("lstm", "gru", "cnn1d", "tabnet", "tab_transformer")


class _Deep(BaseEstimator):
    """One of ARCHS trained with Adam on mini-batches. Every argument is a constructor
    parameter, so scikit-learn's clone / get_params / set_params work."""

    # defaults: hidden 64, layers 2, 100 epochs, batch 256, dropout 0.1 = our starting point; learning rate
    # 1e-3 = published default of Adam (Kingma & Ba 2015; torch.optim.Adam); tol 1e-4 = published default of
    # scikit-learn's MLP (the epoch loop's early-stopping tolerance). TabNet overrides them in _DEFAULTS.
    def __init__(self, arch="lstm", hidden=64, layers=2, max_iter=100, learning_rate=1e-3, batch_size=256,
                 dropout=0.1, class_weight=None, tol=1e-4, random_state=None):
        self.arch, self.hidden, self.layers, self.max_iter = arch, hidden, layers, max_iter
        self.learning_rate, self.batch_size, self.dropout = learning_rate, batch_size, dropout
        self.class_weight, self.tol, self.random_state = class_weight, tol, random_state

    # --- input --------------------------------------------------------------
    def _check(self, X):
        from scipy.sparse import issparse
        if issparse(X):
            raise TypeError(f"{self.arch} needs dense data - text columns give thousands of sparse features.")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("Expected a 2-D table of numbers.")
        if not np.isfinite(X).all():
            raise ValueError(f"{self.arch} can't handle missing values - add an impute step first.")
        if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} columns, got {X.shape[1]}.")
        return X

    def _net(self, n, out):
        if self.arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}.")
        if self.arch != "tabnet" and n > MAX_TOKENS:
            raise ValueError(f"{self.arch} takes at most {MAX_TOKENS} input columns (this data has {n}).")
        args = (n, out, int(self.hidden), int(self.layers), float(self.dropout))
        if self.arch in ("lstm", "gru"):
            return Recurrent(*args, self.arch)
        return {"cnn1d": Conv, "tabnet": TabNet, "tab_transformer": Transformer}[self.arch](*args)

    def _start(self, X, y, classes):
        seed = 0 if self.random_state is None else int(self.random_state)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        self.n_features_in_ = X.shape[1]
        self.x_mean_ = X.mean(axis=0)
        sd = X.std(axis=0)
        self.x_std_ = np.where(sd > 0, sd, 1.0).astype(np.float32)
        out = self._outputs(y, classes)
        self.net_ = self._net(X.shape[1], out).to(device())
        self.opt_ = torch.optim.Adam(self.net_.parameters(), lr=float(self.learning_rate))
        self.gen_ = torch.Generator().manual_seed(seed)
        self.n_iter_ = 0

    def _scaled(self, X):
        return torch.from_numpy((X - self.x_mean_) / self.x_std_).to(device())

    def _on_device(self):
        dev = device()
        if next(self.net_.parameters()).device != dev:
            self.net_.to(dev)
        if not hasattr(self, "opt_"):              # a saved / copied model keeps no optimizer
            self.opt_ = torch.optim.Adam(self.net_.parameters(), lr=float(self.learning_rate))
            self.gen_ = torch.Generator().manual_seed(int(self.random_state or 0) + self.n_iter_)

    # --- training -------------------------------------------------------------
    def partial_fit(self, X, y, classes=None):
        """One epoch over the rows in shuffled mini-batches (initialises on the first call)."""
        X = self._check(X)
        if not hasattr(self, "net_"):
            self._start(X, y, classes)
        self._on_device()
        Xt, yt = self._scaled(X), self._target(y)
        self.net_.train()
        order = torch.randperm(len(X), generator=self.gen_).to(Xt.device)
        batches = list(order.split(max(1, min(int(self.batch_size), len(X)))))
        if len(batches) > 1 and len(batches[-1]) == 1:   # batch norm can't train on a single row
            batches[-2:] = [torch.cat(batches[-2:])]
        total = torch.zeros((), device=Xt.device)
        for idx in batches:
            out, penalty = self.net_(Xt[idx])
            loss = self._loss(out, yt[idx])
            self.opt_.zero_grad()
            (loss + TABNET_SPARSITY * penalty).backward()
            self.opt_.step()
            total += self._shown(out.detach(), yt[idx], loss.detach()) * len(idx)
        self.loss_ = self._report(float(total) / len(X))
        self.n_iter_ += 1
        return self

    def fit(self, X, y):
        for attr in ("net_", "opt_", "gen_", "n_features_in_"):
            self.__dict__.pop(attr, None)
        for _ in range(int(self.max_iter)):
            self.partial_fit(X, y)
        return self

    def _predict_rows(self):
        """Rows per forward pass: attention memory grows with columns squared."""
        if self.arch != "tab_transformer":
            return PREDICT_BATCH
        return int(min(PREDICT_BATCH, max(64, ATTENTION_CELLS // ((self.n_features_in_ + 1) ** 2 * HEADS))))

    def _raw(self, X):
        X = self._check(X)
        self._on_device()
        self.net_.eval()
        outs, step = [], self._predict_rows()
        with torch.no_grad():
            for start in range(0, len(X), step):
                outs.append(self.net_(self._scaled(X[start:start + step]))[0].float().cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, 1), np.float32)

    def __getstate__(self):
        """Saved / copied without the optimizer, weights on the CPU (loads anywhere)."""
        state = {k: v for k, v in self.__dict__.items() if k not in ("opt_", "gen_")}
        if "net_" in state and next(state["net_"].parameters()).is_cuda:
            state["net_"] = copy.deepcopy(state["net_"]).cpu()
        return state


class DeepClassifier(ClassifierMixin, _Deep):
    def _outputs(self, y, classes):
        self.classes_ = np.asarray(classes) if classes is not None else np.unique(y)
        self.weight_ = None
        if self.class_weight == "balanced":        # n / (classes x count), scikit-learn's "balanced"
            counts = np.array([(np.asarray(y) == c).sum() for c in self.classes_], float)
            self.weight_ = len(y) / (len(self.classes_) * np.maximum(counts, 1))
        return len(self.classes_)

    def _target(self, y):
        return torch.from_numpy(np.searchsorted(self.classes_, np.asarray(y)).astype(np.int64)).to(device())

    def _loss(self, out, y):
        w = None if self.weight_ is None else torch.as_tensor(self.weight_, dtype=out.dtype, device=out.device)
        return F.cross_entropy(out, y, weight=w)

    def _shown(self, out, y, loss):
        """The training loss drawn on the epoch chart: plain cross-entropy (= the validation log loss),
        even when class weights shape the loss that is optimised -- else the two curves differ by the
        weighting alone and look like overfitting."""
        return loss if self.weight_ is None else F.cross_entropy(out, y)

    def _report(self, loss):
        return loss                                 # cross-entropy = log loss (the epoch chart's metric)

    def predict_proba(self, X):
        return torch.softmax(torch.from_numpy(self._raw(X)), dim=1).numpy().astype(np.float64)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class DeepRegressor(RegressorMixin, _Deep):
    def _outputs(self, y, classes):
        y = np.asarray(y, float)
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std() or 1.0)
        return 1

    def _target(self, y):
        return torch.from_numpy(((np.asarray(y, float) - self.y_mean_) / self.y_std_).astype(np.float32)[:, None]).to(
            device())

    def _loss(self, out, y):
        return F.mse_loss(out, y)                   # on the standardised target

    def _shown(self, out, y, loss):
        return loss

    def _report(self, loss):
        return 0.5 * loss * self.y_std_ ** 2       # half squared error in target units (the MLP's loss_)

    def predict(self, X):
        return self._raw(X)[:, 0].astype(np.float64) * self.y_std_ + self.y_mean_


# --- model-table rows -----------------------------------------------------------

_ABOUT = {
    "lstm": ("LSTM network (long short-term memory)",
             "Reads the column tokens one after another; input, forget and output gates decide what its memory "
             "keeps. Predicts from the final memory.",
             "Built for sequences (time series, text); here for comparison on tables, as the plan lists it.",
             "Treats column order as meaningful, which in a table it is not. Usually below gradient boosting on tables."),
    "gru": ("GRU network (gated recurrent unit)",
            "Like the LSTM with 2 gates instead of 3 - fewer weights, faster, less overfitting on small data.",
            "Sequences; a lighter recurrent comparison on tables.",
            "Same column-order caveat as the LSTM."),
    "cnn1d": ("1-D convolutional network",
              "Small filters slide over neighbouring column tokens to find local patterns, then are averaged.",
              "Signal-like data where neighbouring columns belong together (sensor readings over time, spectra).",
              "On ordinary tables which columns are neighbours is arbitrary."),
    "tabnet": ("TabNet-style network (Arik & Pfister 2021)",
               "In each decision step a sparse attention mask picks the few columns that step looks at for this "
               "row; the steps' outputs are added up.",
               "Designed for tables; the masks select columns per row.",
               "Written here (pytorch-tabnet is not installed); needs more rows than the trees to shine."),
    "tab_transformer": ("TabTransformer (FT-Transformer style)",
                        "Every column becomes a token; self-attention lets each column look at all the others, and "
                        "a summary token makes the prediction.",
                        "Tables where columns interact; one of the stronger deep models for tables.",
                        f"Memory grows with columns squared - refuses more than {MAX_TOKENS} input columns."),
}
# per-architecture defaults: pytorch-tabnet's published ones for TabNet (n_d = n_a = 8, n_steps = 3,
# Adam lr 0.02, batch 1024); our starting point for the others (Adam's published lr 1e-3)
_DEFAULTS = {"lstm": {"hidden": 64, "layers": 2}, "gru": {"hidden": 64, "layers": 2},
             "cnn1d": {"hidden": 64, "layers": 2},
             "tabnet": {"hidden": 8, "layers": 3, "learning_rate": 0.02, "batch_size": 1024},
             "tab_transformer": {"hidden": 64, "layers": 2}}
_LAYERS = {"lstm": "Recurrent layers", "gru": "Recurrent layers", "cnn1d": "Convolution layers",
           "tabnet": "Decision steps", "tab_transformer": "Transformer blocks"}


def rows(task):
    """Model-table rows (training/models.py shape) for the five architectures."""
    from training.params import fit_specs, spec
    clf = task == "classification"
    est = DeepClassifier if clf else DeepRegressor
    gpu = torch.cuda.is_available()
    out = {}
    for arch in ARCHS:
        what, how, good, watch = _ABOUT[arch]
        params = [spec("hidden", "Hidden size", "int", 4, 1024,
                       help="width of each layer (TabNet: n_d = n_a); larger = more capacity, slower"),
                  spec("layers", _LAYERS[arch], "int", 1, 10, help="depth: more = more capacity, slower, overfits sooner"),
                  spec("max_iter", "Epochs", "int", 1, 10000, help="Epochs: full passes over the training rows."),
                  spec("learning_rate", "Learning rate", "float", 1e-6, 1.0, log=True, help="Adam's step size."),
                  spec("batch_size", "Batch size", "int", 2, 100000,   # batch norm needs 2+ rows
                       help="rows per weight update"),
                  spec("dropout", "Dropout", "float", 0.0, 0.9, help="share of units switched off while training "
                                                                     "(regularisation)")] + fit_specs()
        out[arch] = {"label": what.split(" (")[0] if arch != "tab_transformer" else what, "family": "Deep learning",
                     "make": lambda seed, a=arch: est(arch=a, random_state=seed, **_DEFAULTS[a]),
                     "sparse": False, "nonneg": False, "nan": False,
                     "balanced": {"class_weight": "balanced"} if clf else None,
                     "note": "standardises its inputs itself; text columns are too wide for it",
                     "params": params, "epochs": "mlp", "resolve": None, "device": "gpu" if gpu else "cpu",
                     "max_features": None if arch == "tabnet" else MAX_TOKENS,
                     "about": {"what": what + ".", "how": how, "good": good, "watch": watch}}
    return out


if __name__ == "__main__":
    import io
    import time
    import warnings

    import joblib
    from scipy.sparse import csr_matrix
    from sklearn.base import clone
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    n = 800
    X = rng.normal(size=(n, 8)) * [1, 10, 1, 100, 1, 1, 1, 1]       # unscaled on purpose
    y_c = (X[:, 0] + X[:, 1] / 10 > 0).astype(int)
    y_m = np.digitize(X[:, 0] + X[:, 1] / 10, [-0.6, 0.6])
    y_r = 3 * X[:, 0] + X[:, 1] / 5 + rng.normal(0, .3, n)
    tr, te = slice(0, 600), slice(600, n)
    print("device:", device())
    for arch in ARCHS:
        t0 = time.time()
        d = _DEFAULTS[arch]
        clf = DeepClassifier(arch=arch, max_iter=60, random_state=0, **d).fit(X[tr], y_c[tr])
        acc = (clf.predict(X[te]) == y_c[te]).mean()
        multi = DeepClassifier(arch=arch, max_iter=60, random_state=0, **d).fit(X[tr], y_m[tr])
        macc = (multi.predict(X[te]) == y_m[te]).mean()
        reg = DeepRegressor(arch=arch, max_iter=60, random_state=0, **d).fit(X[tr], y_r[tr])
        r2 = reg.score(X[te], y_r[te])
        print(f"{arch}: binary acc {acc:.3f}, 3-class acc {macc:.3f}, regression R2 {r2:.3f} ({time.time() - t0:.1f}s)")
        assert acc > 0.85 and macc > 0.7 and r2 > 0.8, (arch, acc, macc, r2)
        p = clf.predict_proba(X[te])
        assert p.shape == (200, 2) and np.allclose(p.sum(axis=1), 1, atol=1e-5)
        # saved model: weights on the CPU, same predictions after loading
        buf = io.BytesIO()
        joblib.dump(clf, buf)
        loaded = joblib.load(io.BytesIO(buf.getvalue()))
        assert next(loaded.net_.parameters()).device.type == "cpu"
        assert np.allclose(loaded.predict_proba(X[te]), p, atol=1e-5)
        # same seed, same settings -> same model
        again = DeepClassifier(arch=arch, max_iter=60, random_state=0, **d).fit(X[tr], y_c[tr]).predict_proba(X[te])
        drift = float(np.abs(again - p).max())
        print(f"  same seed twice: max probability difference {drift:.2e}")
        assert drift < 1e-3, (arch, drift)
        # refusals the model table's flags promise (models.incompatible)
        for bad, err in ((csr_matrix(X[:50]), TypeError), (np.where(X[:50] > 2, np.nan, X[:50]), ValueError)):
            try:
                DeepClassifier(arch=arch, max_iter=1).fit(bad, y_c[:50])
                raise AssertionError((arch, "accepted bad input"))
            except err:
                pass
        assert clone(clf).get_params()["arch"] == arch and not hasattr(clone(clf), "net_")
    # class weights reach the loss; partial_fit counts epochs and the loss falls
    w = DeepClassifier(arch="gru", class_weight="balanced", max_iter=2, random_state=0).fit(X[tr], (y_c[tr] * (X[tr, 2] > 1)))
    assert w.weight_[1] > w.weight_[0]
    m = DeepRegressor(arch="tabnet", random_state=0, **_DEFAULTS["tabnet"])
    losses = [m.partial_fit(X[tr], y_r[tr]).loss_ for _ in range(30)]
    assert m.n_iter_ == 30 and losses[-1] < losses[0], losses[:3]
    # sparsemax: rows sum to 1, many exact zeros
    s = sparsemax(torch.tensor([[3.0, 1.0, 0.2, -1.0], [0.1, 0.1, 0.1, 0.1]]))
    assert torch.allclose(s.sum(dim=1), torch.ones(2)) and (s[0] == 0).sum() >= 2 and torch.allclose(s[1], torch.full((4,), .25))
    # wide data: the transformer predicts in small batches (attention ~ columns squared), others in big ones
    wide = DeepClassifier(arch="tab_transformer", max_iter=1, random_state=0).fit(rng.normal(size=(80, 512)),
                                                                                  np.arange(80) % 2)
    assert wide._predict_rows() == 64 and len(wide.predict(rng.normal(size=(150, 512)))) == 150
    assert DeepClassifier(arch="gru", max_iter=1).fit(X[:40], y_c[:40])._predict_rows() == PREDICT_BATCH
    # class weights: the chart's training loss is plain cross-entropy, comparable with the validation log loss
    from sklearn.metrics import log_loss as _ll
    cw = DeepClassifier(arch="cnn1d", class_weight="balanced", random_state=0)
    yy = (X[tr, 0] > 1).astype(int)
    for _ in range(3):
        cw.partial_fit(X[tr], yy, classes=np.array([0, 1]))
    assert abs(cw.loss_ - _ll(yy, cw.predict_proba(X[tr]), labels=[0, 1])) < 0.15, cw.loss_
    try:
        DeepClassifier(arch="cnn1d", max_iter=1).fit(np.zeros((10, MAX_TOKENS + 1)), np.arange(10) % 2)
        raise AssertionError("too wide accepted")
    except ValueError:
        pass

    # end to end through the app's own run: CV + locked test, live epoch curves with accuracy,
    # early stopping on held-back rows, the saved model predicting, LIME explaining it
    import tempfile
    from pathlib import Path

    import pandas as pd
    from core import jobs, store
    from explain import local
    from preprocessing import execute
    from training import models, train
    store.ROOT = Path(tempfile.mkdtemp())
    df = pd.DataFrame(X[:, :6], columns=[f"x{i}" for i in range(6)]).assign(
        color=rng.choice(["red", "blue"], n), y=np.where(y_c == 1, "yes", "no"))
    ds = store.create(df, "t.csv")
    r = execute.apply_plan(df, "y", {"color": [{"op": "encode", "method": "onehot"}]}, task="classification")
    r["fitted"]["clean_ops"] = []
    store.save_prep(ds, r["csv"], r["fitted"])
    listed = {m["key"]: m for m in train.options(ds)["models"]}
    assert all(listed[a]["compatible"] and listed[a]["epochs"] for a in ARCHS), {a: listed[a]["reason"] for a in ARCHS}
    keys = ["lstm", "tabnet", "tab_transformer"]
    s = train.validate(ds, {"models": keys, "folds": 3, "primary_metric": "accuracy", "time_limit_min": 10,
                            "validation_share": 0.2,
                            "params": {k: {"max_iter": 80, "early_stopping": True, "patience": 5} for k in keys}})
    run = jobs.start(ds, lambda job: train.run(job, ds, s), s, 900)
    jobs.wait(run)
    res = store.read_json(store.run_dir(ds, run) / "results.json")
    for k in keys:
        rec = res["models"][k]
        assert "error" not in rec and rec["test"]["accuracy"] > 0.85 and rec["family"] == "Deep learning", (k, rec)
        assert rec["epochs"]["epochs_trained"] <= 80 and rec["device"] == ("gpu" if torch.cuda.is_available() else "cpu")
        curves = store.read_json(store.run_dir(ds, run) / "epochs" / f"{k}.json")
        assert curves["metric"] == "log loss" and curves["accuracy"]["final"], (k, list(curves))
        assert all(v is not None for _e, _t, v in curves["accuracy"]["final"]), k
        e = local.explain_row(ds, run, k, 0)
        assert e["shap"] is None and e["lime"]["columns"][0]["column"] in ("x0", "x1"), (k, e["lime"]["columns"][:3])
    saved = joblib.load(store.run_dir(ds, run) / "models" / "tab_transformer.joblib")
    ctx = train.load_context(ds)
    assert (saved.predict(ctx["X_te"]) == ctx["y_te"]).mean() > 0.85
    # text data: refused with a reason (sparse), like the MLP's other dense-only peers
    facts = {"sparse": True, "negative": True, "nan": False, "features": 40000}
    assert "dense" in models.incompatible(models.table("classification")["lstm"], facts)
    print("deep learning self-check passed")
