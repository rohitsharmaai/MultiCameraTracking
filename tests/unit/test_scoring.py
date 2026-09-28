from src.identity.scoring import compute_score, temporal_plausibility


CFG = {
    "w_body": 0.5,
    "w_face": 0.3,
    "w_time": 0.1,
    "w_camera": 0.1,
    "w_motion": 0.0,
    "body_quality_gate": 0.3,
    "face_quality_gate": 0.4,
}


def test_quality_is_gate_not_bonus():
    weak = compute_score(
        body_cosine=0.1,
        face_cosine=None,
        body_quality=0.9,
        face_quality=0.0,
        temporal=1.0,
        camera=1.0,
        motion=0.5,
        cfg=CFG,
    )
    strong = compute_score(
        body_cosine=0.95,
        face_cosine=None,
        body_quality=0.4,
        face_quality=0.0,
        temporal=1.0,
        camera=1.0,
        motion=0.5,
        cfg=CFG,
    )
    assert strong.S > weak.S


def test_both_quality_failed():
    s = compute_score(
        body_cosine=0.99,
        face_cosine=0.99,
        body_quality=0.01,
        face_quality=0.01,
        temporal=1.0,
        camera=1.0,
        motion=1.0,
        cfg=CFG,
    )
    assert s.both_quality_failed
    assert s.S == 0.0


def test_terms_scaled_unit():
    s = compute_score(
        body_cosine=1.0,
        face_cosine=1.0,
        body_quality=1.0,
        face_quality=1.0,
        temporal=1.0,
        camera=1.0,
        motion=1.0,
        cfg=CFG,
    )
    assert 0.0 <= s.S <= 1.0


def test_temporal_plausibility():
    assert temporal_plausibility(0) == 1.0
    assert temporal_plausibility(-1) == 0.0
    assert temporal_plausibility(90, max_gap=180) == 0.5


def test_orthogonal_body_does_not_get_high_score():
    s = compute_score(
        body_cosine=0.05,
        face_cosine=None,
        body_quality=0.9,
        face_quality=0.0,
        temporal=1.0,
        camera=1.0,
        motion=0.5,
        cfg=CFG,
    )
    assert s.body_similarity == 0.05
    assert s.S < 0.55
