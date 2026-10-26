"""Coverage for MC2 fused all-to-all classification and the new communication
splits (per-communicator roles, dense/expert FSDP AG/RS, a2a, wait attribution).

The profiling sample used while developing this lived outside the repo; these
tests build the same *shape* of data (op names + FSDP module annotations) so the
classification rules stay pinned without shipping a real capture.
"""
from types import SimpleNamespace

import pandas as pd

from llminsight.metrics.core import communication, _fsdp_unit_counts
from llminsight.parser.profile import (
    _normalize_communication,
    comm_meta_from_name,
    comm_type_from_name,
)


# --------------------------------------------------------------------------- #
def test_comm_meta_names():
    cases = {
        "hcom_allGather__612_1_1@5862": ("allGather", "612", "plain", False),
        "hcom_allGather_AicpuKernel_109_47_1@1278": ("allGather", "109", "plain", False),
        "hcom_reduceScatter__612_66_1@5862": ("reduceScatter", "612", "plain", False),
        "AlltoAllvGroupedMatMulMc2AicpuKernel_823_65_1@489": ("alltoallv", "823", "mc2_dispatch", True),
        "GroupedMatMulAlltoAllvMc2AicpuKernel_823_1_1@489": ("alltoallv", "823", "mc2_combine", True),
        "Total Op Info": ("Total", None, "plain", False),
        "some_unknown_kernel_7_1_1@1": ("unknown", "7", "plain", False),
    }
    for name, want in cases.items():
        meta = comm_meta_from_name(name)
        assert (meta["type"], meta["group"], meta["flavor"], meta["fused"]) == want, name
    # historical helper keeps returning just the type token
    assert comm_type_from_name("GroupedMatMulAlltoAllvMc2AicpuKernel_823_1_1@1") == "alltoallv"


def _op(name, elapse, wait, size_mb, transit_ms, sync=0.0):
    return {
        "step": "1", "kind": "collective", "name": name,
        **{k: v for k, v in comm_meta_from_name(name).items() if k != "type"},
        "type": comm_meta_from_name(name)["type"],
        "elapse_ms": elapse, "wait_ms": wait, "sync_ms": sync, "idle_ms": 0.0,
        "transit_ms": transit_ms, "wait_ratio": (wait / elapse if elapse else 0.0),
        "sync_ratio": 0.0, "start_us": 0.0,
        "links": {"UB": {"transit_mb": size_mb, "transit_ms": transit_ms,
                         "bandwidth_gbps": 0.0}},
    }


def _prof(comms, operator_names):
    od = pd.DataFrame({"Name": operator_names})
    return SimpleNamespace(communication=comms, operator_details=od, meta={})


# --------------------------------------------------------------------------- #
def test_normalize_communication_carries_group_and_flavor():
    raw = {
        "step3": {"collective": {
            "GroupedMatMulAlltoAllvMc2AicpuKernel_823_1_1@1": {
                "Communication Time Info": {"Elapse Time(ms)": 3.0, "Transit Time(ms)": 2.0,
                                            "Wait Time(ms)": 0.5},
                "Communication Bandwidth Info": {"UB": {"Transit Size(MB)": 381.0,
                                                        "Transit Time(ms)": 18.0}},
            },
            "Total Op Info": {"Communication Time Info": {}, "Communication Bandwidth Info": {}},
        }, "p2p": {}}
    }
    ops = _normalize_communication(raw)
    fused = next(o for o in ops if o["type"] == "alltoallv")
    assert fused["group"] == "823" and fused["flavor"] == "mc2_combine" and fused["fused"]


def test_fsdp_unit_counts_split_dense_vs_expert():
    names = [
        "FSDP::all_gather (model.layers.0)",
        "FSDP::all_gather_copy_out (model.layers.0)",
        "FSDP::all_gather (model.layers.3.mlp.experts)",
        "FSDP::all_gather_copy_out (model.layers.3.mlp.experts)",
        "FSDP::post_backward_reduce (model.layers.0)",
        "FSDP::post_backward_reduce (model.layers.3.mlp.experts)",
    ]
    units = _fsdp_unit_counts(_prof([], names))
    assert units == {"dense_ag_events": 1, "expert_ag_events": 1,
                     "dense_rs_events": 1, "expert_rs_events": 1}
    # no FSDP annotations (e.g. non-FSDP framework) -> unavailable, not an error
    assert _fsdp_unit_counts(_prof([], ["aclnnMatmul"])) == {}


def test_communication_groups_fsdp_and_a2a():
    comms = [
        # dense FSDP group 612: 2 AG + 2 RS with real payload
        _op("hcom_allGather__612_1_1@1", 10.0, 2.0, 100.0, 4.0),
        _op("hcom_allGather__612_2_1@1", 10.0, 2.0, 100.0, 4.0),
        _op("hcom_reduceScatter__612_1_1@1", 20.0, 10.0, 100.0, 5.0),
        _op("hcom_reduceScatter__612_2_1@1", 20.0, 10.0, 100.0, 5.0),
        # expert FSDP group 399: zero-payload AG/RS
        _op("hcom_allGather__399_1_1@2", 8.0, 3.0, 0.0, 0.0),
        _op("hcom_allGather__399_2_1@2", 8.0, 3.0, 0.0, 0.0),
        _op("hcom_reduceScatter__399_1_1@2", 9.0, 3.0, 0.0, 0.0),
        # EP a2a group 823: fused MC2 all-to-all
        _op("AlltoAllvGroupedMatMulMc2AicpuKernel_823_1_1@3", 4.0, 0.4, 381.0, 2.6),
        _op("GroupedMatMulAlltoAllvMc2AicpuKernel_823_2_1@3", 4.0, 1.2, 381.0, 2.6),
    ]
    od = [
        "FSDP::all_gather (model.layers.0)", "FSDP::all_gather (model.layers.1)",
        "FSDP::all_gather (model.layers.3.mlp.experts)",
        "FSDP::all_gather (model.layers.4.mlp.experts)",
        "FSDP::post_backward_reduce (model.layers.0)",
        "FSDP::post_backward_reduce (model.layers.1)",
        "FSDP::post_backward_reduce (model.layers.3.mlp.experts)",
    ]
    cm = communication(_prof(comms, od))

    groups = {g["group"]: g for g in cm["by_group"]}
    assert groups["612"]["role"] == "dense_fsdp"
    assert groups["399"]["role"] == "expert_fsdp"
    assert groups["823"]["role"] == "ep_a2a"

    fs = cm["fsdp"]
    assert fs["available"]
    assert fs["dense"]["ag"]["count"] == 2 and fs["dense"]["rs"]["count"] == 2
    assert fs["dense"]["ag"]["bandwidth_gbps"] == 25.0          # 200 MB / 8 ms
    assert fs["expert"]["ag"]["size_gb"] == 0.0
    assert fs["expert"]["ag"]["bandwidth_gbps"] is None         # nothing moved

    a2a = cm["a2a"]
    assert a2a["available"] and a2a["count"] == 2
    assert a2a["fused"] is True
    assert a2a["dispatch"]["count"] == 1 and a2a["combine"]["count"] == 1
    assert a2a["bandwidth_gbps"] == 146.5                       # 762 MB / 5.2 ms
    assert a2a["per_op_bw_p50"] == 146.5

    wa = cm["wait_attribution"]
    assert wa["total_wait_ms"] == 34.6
    assert wa["by_role"][0]["role"] == "dense_fsdp"             # 24 ms of wait
    assert wa["top"][0]["group"] == "612"

    zt = cm["zero_transit"]
    assert zt["groups"] == ["399"]
    assert zt["by_group"][0]["role_label"] == "expert-FSDP"

    # type aggregation keeps the historical shape and flags the fused family
    a2a_type = next(t for t in cm["by_type"] if t["type"] == "alltoallv")
    assert a2a_type["fused"] and a2a_type["flavors"] == {"mc2_dispatch": 1, "mc2_combine": 1}


def test_communication_without_fsdp_annotations_still_works():
    comms = [
        _op("hcom_allGather__612_1_1@1", 10.0, 2.0, 100.0, 4.0),
        _op("hcom_reduceScatter__612_1_1@1", 20.0, 10.0, 100.0, 5.0),
    ]
    cm = communication(_prof(comms, []))       # single-card style capture, no FSDP hooks
    assert cm["available"]
    assert cm["fsdp"]["available"] is False
    assert cm["by_group"][0]["role"] == "activation"
    assert cm["a2a"]["available"] is False
