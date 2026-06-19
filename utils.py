import os
import json
import orjson
import pandas as pd
import torch
import torch.nn as nn
import numpy as np


CUDA_VISIBLE_DEVICE: str = str(3)


# =============================================================================
# Configuration & Hyperparameters
# =============================================================================
class Config:
    qwen_embed_dim = 4096

    layout_vocab_size = 1100
    layout_hidden_dim = 512
    layout_num_layers = 4
    layout_num_heads = 8
    max_layout_seq_len = 500

    json_vocab_size = 2000
    json_hidden_dim = 512
    json_num_layers = 4
    json_num_heads = 8
    max_json_seq_len = 1000

    pad_token_id = 0
    sos_token_id = 1
    eos_token_id = 2


# =============================================================================
# Decoders
# =============================================================================
class LayoutDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.condition_proj = nn.Linear(config.qwen_embed_dim, config.layout_hidden_dim)
        self.embedding = nn.Embedding(config.layout_vocab_size, config.layout_hidden_dim)
        self.pos_encoder = nn.Embedding(config.max_layout_seq_len, config.layout_hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.layout_hidden_dim, nhead=config.layout_num_heads,
                                                   batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, config.layout_num_layers)
        self.fc_out = nn.Linear(config.layout_hidden_dim, config.layout_vocab_size)
        self.config = config

    def forward(self, qwen_latent, tgt_seq, tgt_mask=None):
        B, seq_len = tgt_seq.size()
        condition = self.condition_proj(qwen_latent).unsqueeze(1)
        positions = torch.arange(0, seq_len, device=tgt_seq.device).unsqueeze(0).expand(B, seq_len)
        tgt_emb = self.embedding(tgt_seq) + self.pos_encoder(positions)
        output = self.transformer_decoder(tgt_emb, condition, tgt_mask=tgt_mask)
        return self.fc_out(output)

    def generate(self, qwen_latent, max_length=100):
        B = qwen_latent.size(0)
        device = qwen_latent.device
        current_seq = torch.full((B, 1), self.config.sos_token_id, dtype=torch.long, device=device)

        for _ in range(max_length):
            seq_len = current_seq.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
            logits = self(qwen_latent, current_seq, tgt_mask=tgt_mask)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            current_seq = torch.cat([current_seq, next_token], dim=1)
            if (next_token == self.config.eos_token_id).all():
                break
        return current_seq


class StructuralDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.condition_proj = nn.Sequential(
            nn.Linear(config.qwen_embed_dim, config.json_hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.json_hidden_dim * 2, config.json_hidden_dim)
        )
        self.embedding = nn.Embedding(config.json_vocab_size, config.json_hidden_dim)
        self.pos_encoder = nn.Embedding(config.max_json_seq_len, config.json_hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.json_hidden_dim, nhead=config.json_num_heads,
                                                   batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, config.json_num_layers)
        self.fc_out = nn.Linear(config.json_hidden_dim, config.json_vocab_size)
        self.config = config

    def forward(self, qwen_latent, tgt_seq, tgt_mask=None):
        B, seq_len = tgt_seq.size()
        condition = self.condition_proj(qwen_latent).unsqueeze(1)
        positions = torch.arange(0, seq_len, device=tgt_seq.device).unsqueeze(0).expand(B, seq_len)
        tgt_emb = self.embedding(tgt_seq) + self.pos_encoder(positions)
        output = self.transformer_decoder(tgt_emb, condition, tgt_mask=tgt_mask)
        return self.fc_out(output)

    def generate(self, qwen_latent, max_length=200):
        B = qwen_latent.size(0)
        device = qwen_latent.device
        current_seq = torch.full((B, 1), self.config.sos_token_id, dtype=torch.long, device=device)

        for _ in range(max_length):
            seq_len = current_seq.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
            logits = self(qwen_latent, current_seq, tgt_mask=tgt_mask)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            current_seq = torch.cat([current_seq, next_token], dim=1)
            if (next_token == self.config.eos_token_id).all():
                break
        return current_seq


# =============================================================================
# Rico cleaning  (cleaning mechanism ported from cleaning/clean_rico.py)
# -----------------------------------------------------------------------------
# Produces a screen object  {app, activity, components}.  `app` and `activity`
# come from Rico's top-level activity_name and capture the screen's function /
# identity; `components` keeps the *visible* parts of the hierarchy and drops
# invisible layout scaffolding.
#
# Per-node keep-set (everything else is dropped):
#     - class         : full Android class path (e.g. android.widget.Button)
#     - text          : verbatim, when non-empty
#     - content-desc  : accessibility label, only when a real (non-null) string
#     - resource-id   : developer id (e.g. "action_share"), generic ids dropped
#     - clickable     : emitted only when True (presence == clickable)
#     - bounds        : binned to 1-100 against the device frame (root bounds)
#     - children      : nested visible components (grouping preserved)
# =============================================================================

# Containers that actually draw something on screen -> kept as grouping nodes.
VISIBLE_CONTAINER_SUFFIXES = (
    "CardView", "Toolbar", "ToolBar", "ActionBar", "TitleBar",
    "ListView", "RecyclerView", "GridView", "ExpandableListView", "ListPopupWindow",
)
# Inert wrappers that draw nothing -> unwrapped (children lifted up).
SCAFFOLD_EXACT = {
    "View", "ViewGroup", "ScrollView", "HorizontalScrollView", "NestedScrollView",
    "ViewPager", "ViewPager2", "ActionMenuView", "ActionBarContainer", "ActionBarContextView",
}
# Generic framework / system resource-ids that carry no app-specific meaning.
SYSTEM_IDS = {
    "content", "navigationBarBackground", "statusBarBackground", "action_mode_bar_stub",
    "decor_content_parent", "action_bar_container", "action_bar", "action_bar_root",
    "parentPanel", "contentPanel", "topPanel", "customPanel", "buttonPanel", "custom",
}


# -----------------------------------------------------------------------------
# Class-based role detection
# -----------------------------------------------------------------------------
def short_name(node):
    # last segment of the class path: "android.widget.Button" -> "Button"
    cls = node.get("class") or ""
    return cls.split(".")[-1]


def is_visible_container(node):
    # a container that actually draws (Card/Toolbar/List/Grid) -> keep, never unwrap
    s = short_name(node)
    return any(s.endswith(suf) for suf in VISIBLE_CONTAINER_SUFFIXES)


def is_scaffold(node):
    # an inert wrapper that draws nothing -> should be unwrapped (children lifted)
    s = short_name(node)
    if is_visible_container(node):
        return False                 # visible containers are never scaffolding
    if "DecorView" in s:
        return True                  # the top-level window wrapper
    if s.endswith("Layout"):
        return True                  # any *Layout is a grouping container
    return s in SCAFFOLD_EXACT        # other known inert wrappers


def is_bare_view(node):
    # a plain View/ViewGroup, typically a divider / spacer / system-bar strip
    return short_name(node) in ("View", "ViewGroup")


# -----------------------------------------------------------------------------
# Node-level predicates & value extractors
# -----------------------------------------------------------------------------
def is_visible(node):
    """True if the node is shown to the user. Handles both Rico key spellings."""
    v = node.get("visible-to-user", node.get("visible_to_user"))
    if v is None:
        return node.get("visibility") == "visible"
    return bool(v)


def area_ok(bounds):
    """True if bounds is a valid [x1,y1,x2,y2] box with positive area."""
    return (
        isinstance(bounds, list)
        and len(bounds) == 4
        and (bounds[2] - bounds[0]) > 0
        and (bounds[3] - bounds[1]) > 0
    )


def bin_bounds(bounds, width, height):
    """Rescale absolute pixel bounds to 1-100 against the device frame."""
    out = []
    for i, val in enumerate(bounds):
        # even index = x (scale by width), odd index = y (scale by height)
        denom = (width if i % 2 == 0 else height) or 1
        out.append(max(1, min(100, round(val / denom * 100))))
    return out


def text_of(node):
    t = node.get("text")
    if t and str(t).strip():
        return str(t).strip()
    return None


def content_desc_of(node):
    """Flatten Rico's content-desc list to the first real label, or None."""
    cd = node.get("content-desc")
    if isinstance(cd, list):
        cd = next((x for x in cd if x and str(x).strip()), None)
    if cd and str(cd).strip():
        return str(cd).strip()
    return None


def resource_id_of(node):
    """Short developer id (after 'id/'), dropping generic framework/system ids."""
    short = (node.get("resource-id") or "").split("/")[-1]
    if short and short not in SYSTEM_IDS:
        return short
    return None


# -----------------------------------------------------------------------------
# Core cleaning  (returns a LIST of components per node, to allow unwrapping)
# -----------------------------------------------------------------------------
def clean_node(node, width, height):
    """Clean a raw Rico node. Returns a list of cleaned components (0, 1, or many)."""
    # PRUNE: invisible or zero-area nodes (and their whole subtree) are skipped
    if not isinstance(node, dict) or not is_visible(node):
        return []
    bounds = node.get("bounds")
    if not area_ok(bounds):
        return []

    # recurse first: children come back already cleaned/unwrapped/flattened
    kids = []
    for c in node.get("children") or []:
        kids.extend(clean_node(c, width, height))

    clickable = node.get("clickable") is True

    # UNWRAP: inert scaffolding draws nothing -> drop self, lift its children up
    if is_scaffold(node) and not clickable:
        return kids

    # DROP: a bare View with no text/desc/children/clickability is just a strip
    if is_bare_view(node) and not clickable and not text_of(node) and not content_desc_of(node) and not kids:
        return []

    # KEEP: build the component with keep-set fields only, in a stable order
    out = {}
    cls = node.get("class")
    if cls:
        out["class"] = cls
    text = text_of(node)
    if text:
        out["text"] = text
    desc = content_desc_of(node)
    if desc:
        out["content-desc"] = desc
    rid = resource_id_of(node)
    if rid:
        out["resource-id"] = rid
    if clickable:
        out["clickable"] = True
    out["bounds"] = bin_bounds(bounds, width, height)
    if kids:
        out["children"] = kids
    return [out]


def device_frame(root):
    """Device width/height from root bounds; fall back to max observed bounds."""
    b = root.get("bounds")
    if area_ok(b):
        return b[2], b[3]
    max_x = max_y = 1

    def scan(n):
        nonlocal max_x, max_y
        if not isinstance(n, dict):
            return
        bb = n.get("bounds")
        if isinstance(bb, list) and len(bb) == 4:
            max_x = max(max_x, int(bb[2]))
            max_y = max(max_y, int(bb[3]))
        for c in n.get("children") or []:
            scan(c)

    scan(root)
    return max_x, max_y


def parse_activity_name(raw):
    """Split Rico's top-level 'activity_name' into (app package, short activity class).

    e.g. "se.perigee.android.seven/com.perigee.seven.ui.activity.WorkoutSessionCompleteActivity"
         -> ("se.perigee.android.seven", "WorkoutSessionCompleteActivity")
    """
    name = raw.get("activity_name") or ""
    if "/" in name:
        pkg, activity = name.split("/", 1)
    else:
        pkg, activity = "", name
    return pkg, activity.split(".")[-1]


def clean_screen(json_path):
    """Load a raw Rico JSON file and return the cleaned screen object.

    {app, activity, components} -- app/activity carry the screen's function;
    components is the list of visible UI components.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    app, activity = parse_activity_name(raw)          # screen function / identity
    root = raw.get("activity", {}).get("root")
    # clean the view tree against the device frame -> flat-ish forest of components
    components = clean_node(root, *device_frame(root)) if root else []
    return {"app": app, "activity": activity, "components": components}


# =============================================================================
# Shared Functions & Parsers
# =============================================================================
def process_json_file(json_path, minify=True):
    """
    Clean a raw Rico JSON file into the compact screen representation used by
    the pipeline. Returns a JSON string of {app, activity, components}.

    Purely in-memory: reads the input read-only and returns a string; it does
    not write any cleaned files to disk.

        minify=True  -> compact, whitespace-free JSON (default, for embeddings)
        minify=False -> pretty-printed JSON (for inspection)
    """
    assert os.path.exists(json_path), f"Could not find path: {json_path}"
    screen = clean_screen(json_path)
    if minify:
        return _minify_json(screen)
    return json.dumps(screen, indent=2)


def process_json_string(json_string, minify=True):
    """
    Clean a raw Rico JSON string into the compact screen representation used by
    the pipeline. Returns a JSON string of {app, activity, components}.

    Purely in-memory: reads the input read-only and returns a string; it does
    not write any cleaned files to disk.

        minify=True  -> compact, whitespace-free JSON (default, for embeddings)
        minify=False -> pretty-printed JSON (for inspection)
    """
    raw = json.loads(json_string)
    app, activity = parse_activity_name(raw)
    root = raw.get("activity", {}).get("root")
    components = clean_node(root, *device_frame(root)) if root else []
    screen = {"app": app, "activity": activity, "components": components}

    if minify:
        return _minify_json(screen)
    return json.dumps(screen, indent=2)


def _minify_json(json_data):
    """
    Removes all unnecessary whitespaces from a JSON object.
    This creates the most compact valid JSON representation.
    """
    return json.dumps(json_data, separators=(',', ':'))


# ==========================================
# Decoding Stuff
# ==========================================

def _decode_layout_tokens(token_list, config):
    decoded_boxes = []
    tokens = token_list[1:] if token_list and token_list[0] == config.sos_token_id else token_list
    for i in range(0, len(tokens), 5):
        chunk = tokens[i:i + 5]
        if config.eos_token_id in chunk:
            break
        if len(chunk) == 5:
            x1, y1, x2, y2 = [val - 3 for val in chunk[:4]]
            cls_id = chunk[4]
            box = {"class_id": f"UI_Element_{cls_id}", "bbox": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]}
            decoded_boxes.append(box)
    return decoded_boxes


def _decode_json_tokens(token_list, config):
    tokens = token_list[1:] if token_list and token_list[0] == config.sos_token_id else token_list
    elements = []
    for t in tokens:
        if t == config.eos_token_id:
            break
        if t not in [config.pad_token_id]:
            node_type = "container" if t % 2 == 0 else "component"
            elements.append({"node_id": t, "type": node_type})
    mock_tree = {"type": "root", "children": elements}
    return json.dumps(mock_tree, indent=2)


def load_embeddings(embeddings_dir="embeddings", data_dir=None):
    """
    Finds all .npz files in embeddings_dir, locates their matching raw Rico .json
    files, and loads them into a pandas DataFrame.
    """
    import glob

    if not os.path.exists(embeddings_dir):
        raise FileNotFoundError(f"Error: The embeddings directory '{embeddings_dir}' was not found.")

    if data_dir is None:
        # Check standard default paths
        search_dirs = ["../rico-data/data/combined", "data/rico-examples", "rico/examples"]
    elif isinstance(data_dir, (list, tuple)):
        search_dirs = list(data_dir)
    else:
        search_dirs = [data_dir]

    npz_paths = sorted(glob.glob(os.path.join(embeddings_dir, "*.npz")))

    from tqdm import tqdm
    is_main_process = os.environ.get("LOCAL_RANK", "0") == "0"

    records = []
    for npz_path in npz_paths:
        base_name = os.path.basename(npz_path)
        json_name = base_name.replace(".npz", ".json")
        jpg_name = base_name.replace(".npz", ".jpg")

        # Search for the matching JSON file in the directories
        found_json_path = None
        found_jpg_path = None
        for d in search_dirs:
            candidate_json = os.path.join(d, json_name)
            if os.path.exists(candidate_json):
                found_json_path = candidate_json
                candidate_jpg = os.path.join(d, jpg_name)
                if os.path.exists(candidate_jpg):
                    found_jpg_path = candidate_jpg
                else:
                    found_jpg_path = candidate_json.replace(".json", ".jpg")
                break

        if found_json_path:
            records.append({
                "json-path": found_json_path,
                "jpg-path": found_jpg_path,
                "embedding_path": npz_path
            })

    if not records:
        if is_main_process:
            print(f"Warning: No matching JSON files found in {search_dirs} for embeddings in '{embeddings_dir}'.")
        return pd.DataFrame(columns=["json-path", "jpg-path", "embedding"])

    df = pd.DataFrame(records)

    embeddings = []
    for path in tqdm(
        df["embedding_path"],
        desc="Loading embeddings",
        disable=not is_main_process
    ):
        with np.load(path) as f_data:
            embeddings.append(f_data["embedding"].astype(np.float32))

    df["embedding"] = embeddings
    df = df.drop(columns=["embedding_path"])

    if is_main_process:
        print(f"Successfully loaded {len(df)} embeddings.")
    return df



# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Quick demo: clean one example screen and print the result.
    print("--- Minified ---")
    print(process_json_file("data/rico-examples/0.json", minify=True))

    print("\n--- Pretty ---")
    print(process_json_file("data/rico-examples/0.json", minify=False))