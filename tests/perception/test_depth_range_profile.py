"""The range profile's bands are derived from the declared window.

The measurement itself is the P01-C pipeline measured by its own tests; what is
new here is the band derivation, which is the part a reader must be able to
check without re-running a stereo pipeline.
"""

from __future__ import annotations

from embodied.perception.depth_range_profile import band_edges


def test_bands_tile_the_declared_window():
    assert band_edges(0.5, 6.0) == [
        (0.5, 1.5),
        (1.5, 2.5),
        (2.5, 3.5),
        (3.5, 4.5),
        (4.5, 5.5),
        (5.5, 6.0),
    ]


def test_the_last_band_is_cut_at_the_declared_maximum():
    # A window that is not a whole number of bands long must end at z_max, not
    # run past it: the band edge is the declared range, never a rounded one.
    assert band_edges(0.5, 4.0)[-1] == (3.5, 4.0)


def test_widening_the_window_adds_bands_without_moving_any_band_start():
    # The property the R21 re-run depends on: the bands are a function of the
    # declared window, so widening it 4 m -> 6 m leaves every band START where it
    # was and appends. The one band that changes is the last, because it is cut
    # at the window's own maximum (4.0 then, 6.0 now); asserting that the upper
    # edges were preserved too would be asserting something false, which an
    # earlier version of this test did.
    four = band_edges(0.5, 4.0)
    six = band_edges(0.5, 6.0)
    assert [lo for lo, _ in six[: len(four)]] == [lo for lo, _ in four]
    assert four[-1] == (3.5, 4.0)
    assert six[-1] == (5.5, 6.0)
    assert len(six) > len(four)


def test_band_width_is_configurable_and_sub_metre_windows_still_tile():
    assert band_edges(1.0, 3.0, 0.5) == [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)]
