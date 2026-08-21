"""
component_utils.py

Shared building blocks for the GUI *component* prediction task (Screen2Vec
paper, Table 2) -- the component-level counterpart to the screen-level task in
4_evaluate_screen_prediction.py.

Holds three things that scripts 10-13 all need:

  1. Screen2Vec's 26-category GUI class taxonomy (paper Table 1), ported
     verbatim from screen2vec-master/UI_embedding/dataset/convert_class_to_label.py
     so our class labels are the same integers the original model predicted.
  2. Component extraction: flattens the cleaned screen tree produced by
     utils.clean_screen() into an ordered list of components, each carrying
     (text, class label, binned bounds, parent index).
  3. Context selection (EUCLIDEAN and HIERARCHICAL distance, the two rows of
     Table 2) and the ComponentPredictor model, shared by trainer and evaluator
     so the architecture can never drift between them.

Deviation from the original, on purpose:
    screen2vec-master/UI_embedding/dataset/dataset.py::get_closest_UI_obj falls
    back to `[*range(len(labeled_uis))]` when a screen has <= n components,
    which puts the *target itself* into its own context. We always exclude the
    target (see closest_context_indices below), so no target text ever leaks
    into the input.
"""

import gzip
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Screen2Vec's 26 GUI class categories (paper Table 1)
# =============================================================================
# Ported from convert_class_to_label.py. Duplicate keys resolve the same way
# they do there (last one wins: TextView -> 20, CheckedTextView -> 18), which
# the TextView special case in convert_class_to_label() overrides anyway.
CLASS_LABEL_LOOKUP = {
    "AdView": 1, "HtmlBannerWebView": 1, "AdContainer": 1,
    "ActionBarOverlayLayout": 2, "TabLayout": 2, "TabLayout$SlidingTabStrip": 2,
    "TabLayout$TabView": 2, "LinearLayout": 2, "FitWindowsLinearLayout": 2,
    "CustomScreenLinearLayout": 2, "AppBarLayout": 2, "FrameLayout": 2,
    "ContentFrameLayout": 2, "FitWindowsFrameLayout": 2, "NoSaveStateFrameLayout": 2,
    "RelativeLayout": 2, "TableLayout": 2,
    "BottomTagGroupView": 3, "BottomBar": 3,
    "ButtonBar": 4,
    "CardView": 5,
    "CheckBox": 6,
    "DrawerLayout": 7,
    "DatePicker": 8,
    "ImageView": 9,
    "ImageButton": 10, "GlyphView": 10, "AppCompactButton": 10,
    "AppCompactImageButton": 10, "ActionMenuItemView": 10, "ActionMenuItemPresenter": 10,
    "EditText": 11, "SearchBoxView": 11, "AppCompatAutoCompleteTextView": 11,
    "TextInputLayout": 11,
    "ListView": 12, "RecyclerView": 12, "ListPopUpWindow": 12, "tabItem": 12,
    "GridView": 12,
    "MapView": 13,
    "SlidingTab": 14,
    "NumberPicker": 15,
    "Switch": 16, "SwitchCompat": 16,
    "ViewPageIndicatorDots": 17, "PageIndicator": 17, "CircleIndicator": 17,
    "PagerIndicator": 17,
    "RadioButton": 18, "CheckedTextView": 18,
    "SeekBar": 19,
    "Button": 20, "TextView": 20,
    "ToolBar": 21, "Toolbar": 21, "TitleBar": 21, "ActionBar": 21,
    "VideoView": 22,
    "WebView": 23,
    # 24 (in list) and 25 (in drawer) are assigned contextually, not by class.
}

NUM_CLASSES = 26
LIST_CLASS = 12      # descendants of a list get label 24 when otherwise unknown
DRAWER_CLASS = 7     # descendants of a drawer get label 25 when otherwise unknown
IN_LIST_CLASS = 24
IN_DRAWER_CLASS = 25

# The empty string sits at index 0 of the vocabulary and is never a target: the
# task is "predict the text content", so components without text are context
# only. (The original reserves index 0 via CrossEntropyLoss' ignore_index=0 but
# appends '' to the *end* of its vocab list, so it actually ignores whichever
# real string happened to land first. We make it explicit.)
EMPTY_TEXT_INDEX = 0


def convert_class_to_label(class_path, clickable=False):
    """Map an Android class path to one of Screen2Vec's 26 categories.

    Mirrors convert_class_to_text_label() plus the TextView rule from
    rico_utils.get_all_labeled_uis_from_node_tree: a clickable TextView is a
    "Text Button" (20), a non-clickable one is "Input" (11).
    """
    if not class_path or not class_path.strip():
        return 0
    short = class_path.split(".")[-1]
    if short == "TextView":
        return 20 if clickable else 11
    label = CLASS_LABEL_LOOKUP.get(short)
    if label:
        return label
    for key, val in CLASS_LABEL_LOOKUP.items():
        if short.endswith(key) or short.startswith(key):
            return val
    return 0


# =============================================================================
# Component extraction
# =============================================================================
def flatten_components(components, parent=-1, in_list=False, in_drawer=False, out=None):
    """Flatten the nested component tree from utils.clean_screen() into a list.

    Returns a list of dicts in depth-first order:
        {"text": str, "cls": int, "bounds": [x1,y1,x2,y2], "parent": int}
    where `parent` indexes back into the same list (-1 for a root component).
    Bounds are the 1-100 binned coordinates that utils.clean_node() writes.
    """
    if out is None:
        out = []
    for node in components:
        cls = convert_class_to_label(node.get("class"), node.get("clickable") is True)
        # Contextual fallbacks: an otherwise unrecognised component inherits
        # "in list" / "in drawer" from its ancestors (paper Table 1, notes).
        if cls == 0:
            if in_drawer:
                cls = IN_DRAWER_CLASS
            elif in_list:
                cls = IN_LIST_CLASS
        idx = len(out)
        out.append({
            "text": node.get("text") or "",
            "cls": cls,
            "bounds": list(node.get("bounds") or [0, 0, 0, 0]),
            "parent": parent,
        })
        kids = node.get("children")
        if kids:
            flatten_components(
                kids,
                parent=idx,
                in_list=in_list or cls == LIST_CLASS,
                in_drawer=in_drawer or cls == DRAWER_CLASS,
                out=out,
            )
    return out


# =============================================================================
# Context selection: the two distance measures compared in Table 2
# =============================================================================
def euclidean_distance(bounds_a, bounds_b):
    """Screen2Vec's EUCLIDEAN measure between two bounding boxes.

    Ported from RicoScreen.distance_between(): the gap between the closest
    edges in x and y, combined as a euclidean norm.
    """
    x_distance = min(abs(bounds_a[0] - bounds_b[2]), abs(bounds_a[2] - bounds_b[0]))
    y_distance = min(abs(bounds_a[1] - bounds_b[3]), abs(bounds_a[3] - bounds_b[1]))
    return math.sqrt(x_distance * x_distance + y_distance * y_distance)


def hierarchy_distances(parents, target_idx):
    """Screen2Vec's HIERARCHICAL measure: edge count in the view tree.

    BFS from `target_idx` over the undirected parent/child graph. Returns a list
    of distances (inf for components in a disconnected part of the forest, which
    happens because clean_screen() lifts the children of unwrapped scaffolding
    up to the top level).
    """
    n = len(parents)
    children = [[] for _ in range(n)]
    for i, p in enumerate(parents):
        if 0 <= p < n:
            children[p].append(i)

    dist = [math.inf] * n
    dist[target_idx] = 0.0
    queue = [target_idx]
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        neighbours = list(children[cur])
        p = parents[cur]
        if 0 <= p < n:
            neighbours.append(p)
        for nb in neighbours:
            if dist[nb] == math.inf:
                dist[nb] = dist[cur] + 1.0
                queue.append(nb)
    return dist


def closest_context_indices(bounds, parents, target_idx, n, distance="euclidean"):
    """The n components closest to `target_idx`, target always excluded.

    `distance` is "euclidean" (bounding-box gap) or "hierarchical" (view-tree
    edge count) -- the two Screen2Vec variants reported in Table 2. Ties break
    on component index, matching the original's `distances.sort()` over
    [distance, index] pairs.
    """
    if distance == "hierarchical":
        dists = hierarchy_distances(parents, target_idx)
    else:
        target_bounds = bounds[target_idx]
        dists = [euclidean_distance(target_bounds, b) for b in bounds]

    order = sorted(
        (d, i) for i, d in enumerate(dists) if i != target_idx and d != math.inf
    )
    return [i for _, i in order[:n]]


# =============================================================================
# Dataset I/O
# =============================================================================
# One screen per line in components.jsonl.gz:
#   {"id": 12345, "app": "com.foo.bar",
#    "c": [[text_idx, cls, x1, y1, x2, y2, parent], ...]}
COMP_TEXT, COMP_CLS, COMP_X1, COMP_Y1, COMP_X2, COMP_Y2, COMP_PARENT = range(7)


def write_screens(path, screens):
    """Write extracted screens as gzipped JSON lines."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for screen in screens:
            f.write(json.dumps(screen, separators=(",", ":")))
            f.write("\n")


def read_screens(path, limit=None):
    """Read screens written by write_screens()."""
    screens = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            screens.append(json.loads(line))
            if limit is not None and len(screens) >= limit:
                break
    return screens


def load_component_data(data_dir, limit=None):
    """Load {screens, vocab, meta} written by 10_extract_components.py."""
    screens = read_screens(os.path.join(data_dir, "components.jsonl.gz"), limit=limit)
    with open(os.path.join(data_dir, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    meta_path = os.path.join(data_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return screens, vocab, meta


# =============================================================================
# Model
# =============================================================================
class ComponentPredictor(nn.Module):
    """Predicts a GUI component's text embedding from its neighbouring components.

    The same job as Screen2Vec's HiddenLabelPredictorModel (a single Linear over
    the concatenated n context embeddings), with a Transformer encoder in place
    of the concatenation so that the context size can vary and the components
    can attend to each other.

    Each context component enters as one token built from
        text embedding + class embedding + closeness rank (+ spatial features)
    and a learned [CLS] token feeds two heads:
        - text head : a vector in the (optionally PCA-reduced) text embedding
                      space, L2-normalised, ranked against the whole vocabulary
        - class head: logits over the 26 GUI classes (the paper's second loss term)

    Spatial features are the context box, its size, and its offset from the
    target box. Screen2Vec uses positions only to *order* the context, so
    use_spatial=False reproduces its information set; the default adds the
    geometry explicitly.
    """

    SPATIAL_DIM = 8

    def __init__(self, feat_dim, d_model=512, n_layers=4, n_heads=8,
                 num_classes=NUM_CLASSES, max_context=16, use_spatial=True,
                 dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.max_context = max_context
        self.use_spatial = use_spatial
        self.num_classes = num_classes

        self.text_in = nn.Linear(feat_dim, d_model)
        self.class_emb = nn.Embedding(num_classes, d_model)
        self.rank_emb = nn.Embedding(max_context, d_model)
        self.spatial_in = nn.Linear(self.SPATIAL_DIM, d_model) if use_spatial else None
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.in_norm = nn.LayerNorm(d_model)
        self.in_drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.out_norm = nn.LayerNorm(d_model)

        self.text_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, feat_dim),
        )
        self.class_head = nn.Linear(d_model, num_classes)

    def forward(self, ctx_text, ctx_cls, ctx_spatial, ctx_mask):
        """
        Args:
            ctx_text:    (B, T, feat_dim) text embeddings of the context components
            ctx_cls:     (B, T)           class labels of the context components
            ctx_spatial: (B, T, 8)        spatial features (unused if use_spatial=False)
            ctx_mask:    (B, T)           True where the slot holds a real component
        Returns:
            text_pred:    (B, feat_dim) L2-normalised prediction
            class_logits: (B, num_classes)
        """
        B, T, _ = ctx_text.shape
        ranks = torch.arange(T, device=ctx_text.device).clamp_max(self.max_context - 1)
        tokens = self.text_in(ctx_text) + self.class_emb(ctx_cls) + self.rank_emb(ranks)[None]
        if self.spatial_in is not None:
            tokens = tokens + self.spatial_in(ctx_spatial)
        tokens = self.in_drop(self.in_norm(tokens))

        # The [CLS] token is always valid, so a screen whose target has no usable
        # context still produces a finite (if uninformed) prediction.
        tokens = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)
        pad_mask = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=ctx_mask.device), ctx_mask], dim=1
        )
        hidden = self.encoder(tokens, src_key_padding_mask=~pad_mask)
        pooled = self.out_norm(hidden[:, 0])

        text_pred = F.normalize(self.text_head(pooled), p=2, dim=-1)
        return text_pred, self.class_head(pooled)


def build_predictor_from_checkpoint(checkpoint, device="cpu"):
    """Rebuild a ComponentPredictor from a checkpoint written by script 12."""
    model = ComponentPredictor(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


# =============================================================================
# Batching
# =============================================================================
def build_context_tensors(samples, vocab_feats, max_context, use_spatial):
    """Turn a list of samples into the four tensors ComponentPredictor expects.

    A sample is (context_rows, target_row) where each row is the 7-int component
    record described above. Returns (ctx_text, ctx_cls, ctx_spatial, ctx_mask).
    """
    B = len(samples)
    T = max_context
    feat_dim = vocab_feats.shape[1]

    ctx_text = np.zeros((B, T, feat_dim), dtype=np.float32)
    ctx_cls = np.zeros((B, T), dtype=np.int64)
    ctx_spatial = np.zeros((B, T, ComponentPredictor.SPATIAL_DIM), dtype=np.float32)
    ctx_mask = np.zeros((B, T), dtype=bool)

    for b, (context, target) in enumerate(samples):
        tx1, ty1, tx2, ty2 = target[COMP_X1], target[COMP_Y1], target[COMP_X2], target[COMP_Y2]
        tcx, tcy = (tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0
        for t, row in enumerate(context[:T]):
            ctx_text[b, t] = vocab_feats[row[COMP_TEXT]]
            ctx_cls[b, t] = row[COMP_CLS]
            ctx_mask[b, t] = True
            if use_spatial:
                x1, y1, x2, y2 = row[COMP_X1], row[COMP_Y1], row[COMP_X2], row[COMP_Y2]
                ctx_spatial[b, t] = (
                    x1 / 100.0, y1 / 100.0, x2 / 100.0, y2 / 100.0,
                    ((x1 + x2) / 2.0 - tcx) / 100.0, ((y1 + y2) / 2.0 - tcy) / 100.0,
                    (x2 - x1) / 100.0, (y2 - y1) / 100.0,
                )

    return (
        torch.from_numpy(ctx_text),
        torch.from_numpy(ctx_cls),
        torch.from_numpy(ctx_spatial),
        torch.from_numpy(ctx_mask),
    )


# =============================================================================
# Vocabulary projection (PCA)
# =============================================================================
def fit_vocab_pca(vocab_emb, pred_dim, sample=50000, seed=42):
    """Fit a PCA that takes the vocabulary embeddings down to `pred_dim`.

    Prediction and ranking happen in this reduced space, which makes the paper's
    full-vocabulary cross entropy affordable ("each text prediction was compared
    to every possible text embedding in the vocabulary"). Returns
    (mean, components), or (None, None) when pred_dim is 0/None or >= D.
    """
    from sklearn.decomposition import PCA

    if not pred_dim or pred_dim >= vocab_emb.shape[1]:
        return None, None

    rng = np.random.default_rng(seed)
    n = vocab_emb.shape[0]
    if n > sample:
        rows = np.sort(rng.choice(n, size=sample, replace=False))
        fit_on = vocab_emb[rows]
    else:
        fit_on = vocab_emb
    pca = PCA(n_components=pred_dim, svd_solver="randomized", random_state=seed)
    pca.fit(fit_on.astype(np.float32))
    return pca.mean_.astype(np.float32), pca.components_.astype(np.float32)


def apply_vocab_pca(vocab_emb, mean, components, normalize=True):
    """Project embeddings with a PCA from fit_vocab_pca() and L2-normalise."""
    out = vocab_emb.astype(np.float32)
    if components is not None:
        out = (out - mean) @ components.T
    if normalize:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / np.maximum(norms, 1e-12)
    return out.astype(np.float32)
