"""Parse reference_man.yaml into fixed PyTorch tensors for the PBPK ODE system.

The 34-compartment whole-body PBPK graph has:
  - 7 blood pools (venous, arterial, portal_vein, 4 vascular compartments)
  - 15 organ/tissue nodes (11 perfusion-limited + 4 permeability-limited tissues)
  - 8 GI lumen segments
  - 4 sinks (metabolized/excreted)

All tensors use a deterministic alphabetical node ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from torch import Tensor


# Canonical enzyme ordering — fixed across all modules
ENZYME_TAGS: List[str] = [
    "CYP1A2", "CYP2C9", "CYP2D6", "CYP2E1", "CYP3A4",
]

# Tissues that require Kp values (organs with composition data)
# Permeability-limited tissues use lookup_name for drug Kp/PS mapping
KP_TISSUE_NAMES: List[str] = [
    "adipose", "bone", "brain", "gut_wall", "heart", "kidney",
    "liver", "lung", "muscle", "pancreas", "reproductive",
    "rest", "skin", "spleen", "thymus",
]


@dataclass
class PBPKTopology:
    """Immutable container for the 34-compartment PBPK graph topology.

    All tensors are float32 and can be moved to GPU via `to(device)`.
    Node indices follow alphabetical ordering of node names.
    """

    n_nodes: int
    node_names: List[str]
    node_name_to_idx: Dict[str, int]

    # --- Node properties ---
    volumes: Tensor                     # [34] litres
    is_blood_pool: Tensor               # [34] bool
    is_organ: Tensor                    # [34] bool (includes barrier_organ)
    is_lumen: Tensor                    # [34] bool
    is_sink: Tensor                     # [34] bool

    # Tissue composition for Kp calculation (0 for non-organ nodes)
    tissue_fn: Tensor                   # [34] neutral lipid fraction
    tissue_fp: Tensor                   # [34] phospholipid fraction
    tissue_fw: Tensor                   # [34] water fraction

    # --- Flow edges (perfusion transport) ---
    flow_src: Tensor                    # [n_flow] int64
    flow_tgt: Tensor                    # [n_flow] int64
    flow_rates: Tensor                  # [n_flow] L/h

    # --- Diffusion edges (permeability-limited) ---
    diff_vasc_idx: Tensor               # [n_diff] int64
    diff_tissue_idx: Tensor             # [n_diff] int64
    diff_ps_default: Tensor             # [n_diff] L/h (default PS from YAML)
    diff_lookup_names: List[str]        # tissue name for drug PS override

    # --- GI transit edges ---
    transit_src: Tensor                 # [n_transit] int64
    transit_tgt: Tensor                 # [n_transit] int64
    transit_rates: Tensor               # [n_transit] h^-1

    # --- Absorption edges (lumen → gut_wall) ---
    absorb_src: Tensor                  # [n_absorb] int64
    absorb_tgt: Tensor                  # [n_absorb] int64 (all point to gut_wall)
    absorb_ka_fractions: Tensor         # [n_absorb] dimensionless

    # --- Clearance edges ---
    clear_src: Tensor                   # [n_clear] int64
    clear_tgt: Tensor                   # [n_clear] int64
    clear_models: List[str]             # "well_stirred" or "gfr_filtration"

    # --- Enzyme data ---
    enzyme_tags: List[str]              # ordered enzyme names
    enzyme_abundances: Tensor           # [n_enzyme_nodes, n_enzymes] pmol
    enzyme_node_indices: Tensor         # [n_enzyme_nodes] int64
    ivive_scalings: Tensor              # [n_enzyme_nodes] scaling factor

    # --- Kp tissue mapping ---
    kp_tissue_names: List[str]          # 15 tissue names (alphabetical)
    kp_tissue_node_indices: Tensor      # [15] int64 — node index for each Kp tissue

    # --- Special node indices ---
    venous_idx: int
    arterial_idx: int
    portal_vein_idx: int
    stomach_idx: int
    gut_wall_idx: int
    kidney_idx: int

    # --- Global params ---
    cardiac_output: float               # L/h

    def to(self, device: torch.device) -> PBPKTopology:
        """Move all tensors to device. Returns self for chaining."""
        tensor_fields = [
            "volumes", "is_blood_pool", "is_organ", "is_lumen", "is_sink",
            "tissue_fn", "tissue_fp", "tissue_fw",
            "flow_src", "flow_tgt", "flow_rates",
            "diff_vasc_idx", "diff_tissue_idx", "diff_ps_default",
            "transit_src", "transit_tgt", "transit_rates",
            "absorb_src", "absorb_tgt", "absorb_ka_fractions",
            "clear_src", "clear_tgt",
            "enzyme_abundances", "enzyme_node_indices", "ivive_scalings",
            "kp_tissue_node_indices",
        ]
        for fname in tensor_fields:
            val = getattr(self, fname)
            if isinstance(val, Tensor):
                object.__setattr__(self, fname, val.to(device))
        return self


def load_topology(yaml_path: str | Path) -> PBPKTopology:
    """Parse reference_man.yaml and return a PBPKTopology with all tensors resolved.

    Args:
        yaml_path: Path to the YAML file defining the PBPK graph.

    Returns:
        PBPKTopology with deterministic alphabetical node ordering.
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    gp = cfg["global_params"]
    co_mean = gp["cardiac_output"]["mean"]  # 390.0 L/h

    # ── Build sorted node index ──────────────────────────────────
    nodes_raw = cfg["nodes"]
    node_names = sorted(n["name"] for n in nodes_raw)
    name2idx = {name: i for i, name in enumerate(node_names)}
    n_nodes = len(node_names)
    assert n_nodes == 34, f"Expected 34 nodes, got {n_nodes}"

    # Build lookup: name → raw node dict
    node_dict = {n["name"]: n for n in nodes_raw}

    # ── Node properties ──────────────────────────────────────────
    volumes = torch.zeros(n_nodes, dtype=torch.float32)
    is_blood = torch.zeros(n_nodes, dtype=torch.bool)
    is_organ = torch.zeros(n_nodes, dtype=torch.bool)
    is_lumen = torch.zeros(n_nodes, dtype=torch.bool)
    is_sink = torch.zeros(n_nodes, dtype=torch.bool)
    t_fn = torch.zeros(n_nodes, dtype=torch.float32)
    t_fp = torch.zeros(n_nodes, dtype=torch.float32)
    t_fw = torch.zeros(n_nodes, dtype=torch.float32)

    for name in node_names:
        idx = name2idx[name]
        nd = node_dict[name]
        volumes[idx] = nd["volume"]
        ntype = nd["type"]
        if ntype == "blood_pool":
            is_blood[idx] = True
        elif ntype in ("organ", "barrier_organ"):
            is_organ[idx] = True
            comp = nd.get("composition", {})
            t_fn[idx] = comp.get("fn", 0.0)
            t_fp[idx] = comp.get("fp", 0.0)
            t_fw[idx] = comp.get("fw", 0.0)
        elif ntype == "lumen":
            is_lumen[idx] = True
        elif ntype == "sink":
            is_sink[idx] = True

    # ── Enzyme data ──────────────────────────────────────────────
    enzyme_node_list: List[int] = []
    enzyme_abund_list: List[List[float]] = []
    ivive_list: List[float] = []
    global_ivive = gp.get("ivive_scaling", 6e-5)

    for name in node_names:
        nd = node_dict[name]
        if "enzymes" in nd:
            enzyme_node_list.append(name2idx[name])
            row = [float(nd["enzymes"].get(tag, 0.0)) for tag in ENZYME_TAGS]
            enzyme_abund_list.append(row)
            ivive_list.append(nd.get("ivive_scaling", global_ivive))

    enzyme_node_indices = torch.tensor(enzyme_node_list, dtype=torch.long)
    enzyme_abundances = torch.tensor(enzyme_abund_list, dtype=torch.float32)
    ivive_scalings = torch.tensor(ivive_list, dtype=torch.float32)

    # ── Kp tissue mapping ────────────────────────────────────────
    # Map each of the 15 KP_TISSUE_NAMES to its node index.
    # For permeability-limited tissues, use the *_tissue node (has composition).
    # lookup_name in YAML maps vascular/tissue nodes to their parent tissue.
    kp_node_indices: List[int] = []
    for tissue_name in KP_TISSUE_NAMES:
        # Try direct match first (e.g., "brain" → brain)
        if tissue_name in name2idx:
            kp_node_indices.append(name2idx[tissue_name])
        else:
            # Permeability-limited: use {tissue_name}_tissue node
            tissue_node = f"{tissue_name}_tissue"
            assert tissue_node in name2idx, f"No node for Kp tissue '{tissue_name}'"
            kp_node_indices.append(name2idx[tissue_node])
    kp_tissue_node_indices = torch.tensor(kp_node_indices, dtype=torch.long)

    # ── Parse edges ──────────────────────────────────────────────
    edges_raw = cfg["edges"]

    flow_src_l, flow_tgt_l, flow_rate_l = [], [], []
    diff_vasc_l, diff_tis_l, diff_ps_l = [], [], []
    diff_names_l: List[str] = []
    tran_src_l, tran_tgt_l, tran_rate_l = [], [], []
    abs_src_l, abs_tgt_l, abs_ka_l = [], [], []
    clr_src_l, clr_tgt_l = [], []
    clr_models_l: List[str] = []

    # First pass: collect edges with explicit flow_fraction
    inflow_to_node: Dict[str, float] = {}  # node_name → total inflow (L/h)

    for e in edges_raw:
        etype = e["type"]
        src_name = e["source"]
        tgt_name = e["target"]

        if etype == "flow":
            ff = e.get("flow_fraction")
            if ff is not None:
                rate = co_mean * ff
                flow_src_l.append(name2idx[src_name])
                flow_tgt_l.append(name2idx[tgt_name])
                flow_rate_l.append(rate)
                # Track inflow for return-flow resolution
                inflow_to_node[tgt_name] = inflow_to_node.get(tgt_name, 0.0) + rate
            # Edges without flow_fraction are deferred to second pass

        elif etype == "diffusion":
            vasc_name = src_name
            tis_name = tgt_name
            diff_vasc_l.append(name2idx[vasc_name])
            diff_tis_l.append(name2idx[tis_name])
            diff_ps_l.append(e["ps_product"])
            # lookup_name comes from the tissue node
            ln = node_dict[tis_name].get("lookup_name", tis_name)
            diff_names_l.append(ln)

        elif etype == "transit":
            tran_src_l.append(name2idx[src_name])
            tran_tgt_l.append(name2idx[tgt_name])
            tran_rate_l.append(e["transit_rate"])

        elif etype == "absorption":
            abs_src_l.append(name2idx[src_name])
            abs_tgt_l.append(name2idx[tgt_name])
            abs_ka_l.append(e["ka_fraction"])

        elif etype == "clearance":
            clr_src_l.append(name2idx[src_name])
            clr_tgt_l.append(name2idx[tgt_name])
            clr_models_l.append(e["model"])

    # Second pass: resolve return flows without explicit flow_fraction
    for e in edges_raw:
        if e["type"] == "flow" and "flow_fraction" not in e:
            src_name = e["source"]
            tgt_name = e["target"]
            # Return flow rate = total inflow to source node
            rate = inflow_to_node.get(src_name, 0.0)
            assert rate > 0, f"Cannot resolve return flow from {src_name}: no inflow recorded"
            flow_src_l.append(name2idx[src_name])
            flow_tgt_l.append(name2idx[tgt_name])
            flow_rate_l.append(rate)

    # ── Build tensors ────────────────────────────────────────────
    return PBPKTopology(
        n_nodes=n_nodes,
        node_names=node_names,
        node_name_to_idx=name2idx,
        volumes=volumes,
        is_blood_pool=is_blood,
        is_organ=is_organ,
        is_lumen=is_lumen,
        is_sink=is_sink,
        tissue_fn=t_fn,
        tissue_fp=t_fp,
        tissue_fw=t_fw,
        flow_src=torch.tensor(flow_src_l, dtype=torch.long),
        flow_tgt=torch.tensor(flow_tgt_l, dtype=torch.long),
        flow_rates=torch.tensor(flow_rate_l, dtype=torch.float32),
        diff_vasc_idx=torch.tensor(diff_vasc_l, dtype=torch.long),
        diff_tissue_idx=torch.tensor(diff_tis_l, dtype=torch.long),
        diff_ps_default=torch.tensor(diff_ps_l, dtype=torch.float32),
        diff_lookup_names=diff_names_l,
        transit_src=torch.tensor(tran_src_l, dtype=torch.long),
        transit_tgt=torch.tensor(tran_tgt_l, dtype=torch.long),
        transit_rates=torch.tensor(tran_rate_l, dtype=torch.float32),
        absorb_src=torch.tensor(abs_src_l, dtype=torch.long),
        absorb_tgt=torch.tensor(abs_tgt_l, dtype=torch.long),
        absorb_ka_fractions=torch.tensor(abs_ka_l, dtype=torch.float32),
        clear_src=torch.tensor(clr_src_l, dtype=torch.long),
        clear_tgt=torch.tensor(clr_tgt_l, dtype=torch.long),
        clear_models=clr_models_l,
        enzyme_tags=ENZYME_TAGS,
        enzyme_abundances=enzyme_abundances,
        enzyme_node_indices=enzyme_node_indices,
        ivive_scalings=ivive_scalings,
        kp_tissue_names=KP_TISSUE_NAMES,
        kp_tissue_node_indices=kp_tissue_node_indices,
        venous_idx=name2idx["venous_blood"],
        arterial_idx=name2idx["arterial_blood"],
        portal_vein_idx=name2idx["portal_vein"],
        stomach_idx=name2idx["stomach_lumen"],
        gut_wall_idx=name2idx["gut_wall"],
        kidney_idx=name2idx["kidney"],
        cardiac_output=co_mean,
    )
