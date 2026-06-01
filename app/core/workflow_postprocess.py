"""
app/core/workflow_postprocess.py
─────────────────────────────────────────────────────────────────────────────
Workflow Graph Post-Processor

Spec section 4:
   "Agentic process workflow section now includes both a Start Node and
    an End Node in the workflow graph UI and the complete workflow graph
    UI has also been improved to ensure all nodes, edge arrows, lanes,
    connections, and workflow elements are fully visible and clearly
    structured."

The frontend swimlane diagram supports node `type: 'start'` and
`type: 'end'` (rendered as green pills).  The existing backend graph
builder (analysis_service.get_react_flow_data) sometimes returns flows
WITHOUT a Start or End node, depending on which fallback path runs.

This module guarantees both nodes exist on every shape we ship to the
client, so the UI and the exported PDF/DOCX/PPTX (which uses the same
data) always render the canonical structure.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LANE-BASED SHAPE  ({lanes:[{id,label,nodes:[{id,type,label,column}]}],
#                         flow:[{from,to,label?}]})
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_start_end_lane_based(flow_data: Dict[str, Any]) -> Dict[str, Any]:
    lanes = flow_data.get("lanes") or []
    flow  = flow_data.get("flow") or []

    if not lanes:
        # Nothing to do — caller will route into the sequential fallback path.
        return flow_data

    # Collect existing node ids and types
    all_nodes: List[Dict[str, Any]] = []
    for lane in lanes:
        for n in (lane.get("nodes") or []):
            all_nodes.append(n)
    id_set    = {n.get("id") for n in all_nodes if n.get("id")}
    type_set  = {(n.get("type") or "").lower() for n in all_nodes}
    has_start = "start" in type_set
    has_end   = "end"   in type_set

    if has_start and has_end:
        return flow_data  # already canonical

    # Determine column range to place start/end
    cols = [n.get("column", 1) for n in all_nodes if isinstance(n.get("column"), (int, float))]
    min_col = min(cols) if cols else 1
    max_col = max(cols) if cols else 1

    # Identify entry node (no inbound edges) and exit node (no outbound edges).
    inbound:  Dict[str, int] = {}
    outbound: Dict[str, int] = {}
    for e in flow:
        if not isinstance(e, dict):
            continue
        f = e.get("from"); t = e.get("to")
        if t: inbound[t]  = inbound.get(t, 0) + 1
        if f: outbound[f] = outbound.get(f, 0) + 1

    entry_candidates = [n for n in all_nodes
                        if n.get("id") and inbound.get(n["id"], 0) == 0]
    exit_candidates  = [n for n in all_nodes
                        if n.get("id") and outbound.get(n["id"], 0) == 0]

    # First entry node → first node in first lane if no clear winner
    entry_node = entry_candidates[0] if entry_candidates else (
        (lanes[0].get("nodes") or [None])[0]
    )
    exit_node  = exit_candidates[-1] if exit_candidates else (
        (lanes[-1].get("nodes") or [None])[-1]
    )

    # ── Inject Start node into the first lane ──────────────────────────────
    if not has_start:
        start_id = "__start__"
        # Ensure no id collision
        i = 1
        while start_id in id_set:
            start_id = f"__start_{i}__"
            i += 1
        start_node = {
            "id":     start_id,
            "type":   "start",
            "label":  "Start",
            "column": min_col - 1 if min_col > 0 else 0,
        }
        first_lane = lanes[0]
        first_lane.setdefault("nodes", []).insert(0, start_node)
        id_set.add(start_id)
        if entry_node and entry_node.get("id"):
            flow.insert(0, {"from": start_id, "to": entry_node["id"]})

    # ── Inject End node into the last lane ─────────────────────────────────
    if not has_end:
        end_id = "__end__"
        i = 1
        while end_id in id_set:
            end_id = f"__end_{i}__"
            i += 1
        end_node = {
            "id":     end_id,
            "type":   "end",
            "label":  "End",
            "column": max_col + 1,
        }
        last_lane = lanes[-1]
        last_lane.setdefault("nodes", []).append(end_node)
        id_set.add(end_id)
        if exit_node and exit_node.get("id"):
            flow.append({"from": exit_node["id"], "to": end_id})

    flow_data["lanes"] = lanes
    flow_data["flow"]  = flow
    return flow_data


# ─────────────────────────────────────────────────────────────────────────────
# 2.  REACT-FLOW SHAPE  ({nodes:[{id,type,position,data,...}], edges:[...]})
#     Used by the old sequential-fallback path in analysis_service.
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_start_end_react_flow(flow_data: Dict[str, Any]) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = flow_data.get("nodes") or []
    edges: List[Dict[str, Any]] = flow_data.get("edges") or []

    id_set   = {n.get("id") for n in nodes if n.get("id")}
    type_set = {(n.get("type") or "").lower() for n in nodes}

    has_start = "start" in type_set or "__start__" in id_set
    has_end   = "end"   in type_set or "__end__"   in id_set

    if has_start and has_end:
        return flow_data

    inbound:  Dict[str, int] = {}
    outbound: Dict[str, int] = {}
    for e in edges:
        if not isinstance(e, dict):
            continue
        s = e.get("source"); t = e.get("target")
        if t: inbound[t]  = inbound.get(t, 0) + 1
        if s: outbound[s] = outbound.get(s, 0) + 1

    # Pick entry/exit among non-group nodes
    non_group = [n for n in nodes
                 if n.get("type") not in ("agentGroupNode", "group")]

    entry_candidates = [n for n in non_group
                        if n.get("id") and inbound.get(n["id"], 0) == 0]
    exit_candidates  = [n for n in non_group
                        if n.get("id") and outbound.get(n["id"], 0) == 0]

    entry_node = entry_candidates[0] if entry_candidates else (non_group[0] if non_group else None)
    exit_node  = exit_candidates[-1] if exit_candidates else (non_group[-1] if non_group else None)

    if not has_start and entry_node:
        start_id = "__start__"
        while start_id in id_set:
            start_id += "_"
        nodes.insert(0, {
            "id":   start_id,
            "type": "start",
            "position": {
                "x": (entry_node.get("position", {}).get("x", 60)) - 200,
                "y": entry_node.get("position", {}).get("y", 60),
            },
            "style": {"width": 120, "zIndex": 2},
            "data":  {"label": "Start"},
        })
        id_set.add(start_id)
        edges.insert(0, {
            "id":     f"e-start-{entry_node['id']}",
            "source": start_id,
            "target": entry_node["id"],
            "type":   "smoothstep",
            "animated": True,
            "style":  {"stroke": "#10B981", "strokeWidth": 2},
        })

    if not has_end and exit_node:
        end_id = "__end__"
        while end_id in id_set:
            end_id += "_"
        nodes.append({
            "id":   end_id,
            "type": "end",
            "position": {
                "x": (exit_node.get("position", {}).get("x", 60)) + 400,
                "y": exit_node.get("position", {}).get("y", 60),
            },
            "style": {"width": 120, "zIndex": 2},
            "data":  {"label": "End"},
        })
        id_set.add(end_id)
        edges.append({
            "id":     f"e-{exit_node['id']}-end",
            "source": exit_node["id"],
            "target": end_id,
            "type":   "smoothstep",
            "animated": True,
            "style":  {"stroke": "#10B981", "strokeWidth": 2},
        })

    flow_data["nodes"] = nodes
    flow_data["edges"] = edges
    return flow_data


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def ensure_start_end_nodes(flow_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detects the graph shape and dispatches.  Safe to call on either:
      - lane-based:  has `lanes` + `flow`
      - react-flow:  has `nodes` + `edges`
    Returns the (possibly modified) flow_data dict.
    """
    if not isinstance(flow_data, dict):
        return flow_data
    try:
        if "lanes" in flow_data and "flow" in flow_data:
            return _ensure_start_end_lane_based(flow_data)
        if "nodes" in flow_data and "edges" in flow_data:
            return _ensure_start_end_react_flow(flow_data)
    except Exception as e:
        logger.warning(f"[workflow-postprocess] failed: {e}", exc_info=True)
    return flow_data
