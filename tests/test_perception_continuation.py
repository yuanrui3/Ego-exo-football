import copy

import numpy as np
import pytest

from src.backbone.perception_continuation import (
    appearance_descriptor, associate_focused, prediction_config_hash, preservation_pins,
    same_detections, tracker_eligible, valid_cache,
)


def detection(x,y=100):
    return {"bbox_xyxy_px":[x-3,y-5,x+3,y],"footpoint_xy_px":[x,y],"detector_confidence":.9}


@pytest.mark.parametrize("method",["baseline","damped_recent","damped_appearance"])
def test_trackers_preserve_detections_and_never_fill_missing_samples(method):
    saved=[{"frame_index":i*5,"detections":[detection(i*3)] if i!=4 else []} for i in range(15)]
    immutable=copy.deepcopy(saved)
    feature=[[[1.]+[0.]*31] if row["detections"] else [] for row in saved]
    result=associate_focused(saved,4096,method,feature)
    same_detections(saved,result)
    assert saved==immutable
    assert result[4]["detections"]==[]
    assert len({row["detections"][0]["track"] for row in result if row["detections"]})==1


def test_causal_prefix_invariance_and_empty_frames():
    saved=[{"frame_index":i*5,"detections":[detection(i*2),detection(200-i*2)]} for i in range(20)]
    for method in ("damped_recent","damped_appearance"):
        full=associate_focused(saved,4096,method)
        prefix=associate_focused(saved[:8],4096,method)
        assert prefix==full[:8]
    assert associate_focused([],4096,"damped_recent")==[]


def test_appearance_descriptor_is_video_only_and_normalized():
    image=np.full((150,200,3),(20,30,200),dtype=np.uint8)
    feature=appearance_descriptor(image,[100,100],4096)
    assert len(feature)==32 and np.isclose(sum(feature),1)
    assert appearance_descriptor(image,[-100,-100],4096) is None


def metric(rate=.1,history=.4,fragments=10):
    return {"thresholds":{"40":{"switch_rate":rate,
        "truth_instance_complete_history_fraction":history,
        "fragmentation_extra_tracks_per_identity":fragments}}}


def test_selection_rejects_fragmenting_tracks_or_lost_history():
    old={match:metric() for match in ("117093","117092")}
    good={match:metric(.08,.41,10) for match in old}
    assert tracker_eligible(old,good)
    bad=copy.deepcopy(good)
    bad["117092"]=metric(.01,.1,10)
    assert not tracker_eligible(old,bad)
    bad["117092"]=metric(.01,.41,30)
    assert not tracker_eligible(old,bad)
    assert not tracker_eligible(old,old)


def test_detection_parity_fails_if_a_box_is_changed():
    saved=[{"frame_index":0,"detections":[detection(100)]}]
    altered=copy.deepcopy(saved)
    altered[0]["detections"][0]["footpoint_xy_px"][0]+=1
    with pytest.raises(ValueError,match="changed an observed"):
        same_detections(saved,altered)


def test_cache_requires_checkpoint_input_and_configuration_hashes(tmp_path):
    from src.backbone.gpu_perception import ROOT
    from src.trajectory_dataset import file_digest
    weights=tmp_path/"weights.pt"
    weights.write_bytes(b"disposable")
    stream=tmp_path/"predictions.jsonl"
    stream.write_text("{}\n")
    source={"video_sha256":"video"}
    record={"path":str(stream),"detector_sha256":file_digest(weights),
        "source_video_sha256":"video","configuration_sha256":prediction_config_hash("consistent"),
        "scheme":"consistent","sampling_hz":5,"prediction_jsonl_sha256":file_digest(stream)}
    assert (ROOT/record["path"])==stream
    assert valid_cache(record,weights,source,"consistent")
    for key in ("detector_sha256","source_video_sha256","configuration_sha256","prediction_jsonl_sha256"):
        changed={**record,key:"mismatch"}
        assert not valid_cache(changed,weights,source,"consistent")


def test_manifest_protects_sources_without_requiring_unavailable_source_calibration():
    pins=preservation_pins()
    assert pins["data/117093_1st.json"]=="ceb740ea03b445e27d45f49e7001795f03dbd95eaebde85304e42fd9d0581b19"
    assert sum("keypoints.json" in name for name in pins)==2


@pytest.mark.parametrize("count",[600,1500,3000,3001])
def test_continuous_adapter_reads_are_bounded_without_gaps(count):
    from src.backbone.video_player_cross_match_replay import external_frames
    class Source:
        def frames(self,start,count,stride):
            assert 0<count<=1500 and stride==5
            yield from range(start,start+count*stride,stride)
    assert list(external_frames(Source(),15000,count))==list(range(15000,15000+count*5,5))
