from src.identity.topology import CameraTopology


def test_plausible_known_edge():
    topo = CameraTopology(
        {
            "edges": [
                {
                    "from_camera": "A",
                    "to_camera": "B",
                    "min_transition_seconds": 2,
                    "max_transition_seconds": 10,
                    "expected_direction": "east",
                    "confidence": 0.9,
                }
            ]
        }
    )
    assert topo.is_plausible("A", "B", 5)
    assert not topo.is_plausible("A", "B", 0.5)
    assert not topo.is_plausible("A", "B", 40)
    assert not topo.is_plausible("A", "C", 5)


def test_same_camera_always_ok():
    topo = CameraTopology({"edges": []})
    assert topo.is_plausible("A", "A", 3)
    assert topo.camera_plausibility("A", "A", 3) == 1.0


def test_unknown_edge_is_zero():
    topo = CameraTopology({"edges": []})
    assert topo.camera_plausibility("A", "Z", 5) == 0.0


def test_transition_window():
    topo = CameraTopology(
        {
            "edges": [
                {
                    "from_camera": "A",
                    "to_camera": "B",
                    "min_transition_seconds": 2,
                    "max_transition_seconds": 8,
                    "expected_direction": "east",
                    "confidence": 1.0,
                }
            ]
        }
    )
    inside = topo.camera_plausibility("A", "B", 5)
    outside = topo.camera_plausibility("A", "B", 20)
    assert inside > 0.5
    assert outside == 0.0
