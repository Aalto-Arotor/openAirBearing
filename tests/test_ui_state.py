import numpy as np
import pytest

from openairbearing.app.ui_state import (
    FORM_FIELD_ORDER,
    bearing_to_form_values,
    form_to_bearing_kwargs,
)
from openairbearing.bearings import CircularBearing, JournalBearing


def test_bearing_to_form_values_units():
    bearing = CircularBearing()
    values = bearing_to_form_values(bearing)

    assert values["hp"] == pytest.approx(bearing.hp * 1e3)
    assert values["xa"] == pytest.approx(bearing.xa * 1e3)
    assert values["pa"] == pytest.approx(bearing.pa * 1e-6)
    assert values["ha_min"] == pytest.approx(bearing.ha_min * 1e6)
    assert values["error"] == pytest.approx(bearing.error * 1e6)
    assert values["psi"] == pytest.approx(bearing.Psi)
    assert values["bore_diameter"] is None
    assert values["shaft_diameter"] is None
    assert values["clearance"] is None


def test_form_to_bearing_kwargs_units_and_types():
    bearing = CircularBearing()
    values = bearing_to_form_values(bearing)
    form_values = {
        **values,
        "error_type": "linear",
    }

    kwargs = form_to_bearing_kwargs(form_values)

    assert kwargs["hp"] == pytest.approx(bearing.hp)
    assert kwargs["xa"] == pytest.approx(bearing.xa)
    assert kwargs["pa"] == pytest.approx(bearing.pa)
    assert kwargs["ha_min"] == pytest.approx(bearing.ha_min)
    assert kwargs["error"] == pytest.approx(bearing.error)
    assert kwargs["Psi"] == pytest.approx(bearing.Psi)
    assert isinstance(kwargs["nx"], int)
    assert isinstance(kwargs["ny"], int)
    assert isinstance(kwargs["nh"], int)
    assert np.allclose(kwargs["u"], [values["ux"], values["uy"]])


def test_journal_form_roundtrip_uses_diameters_for_clearance():
    bearing = JournalBearing(bore_diameter=80e-3, shaft_diameter=79.9e-3, nh=5)
    values = bearing_to_form_values(bearing)
    kwargs = form_to_bearing_kwargs({**values, "error_type": "none"}, case="journal")

    assert values["bore_diameter"] == pytest.approx(80.0)
    assert values["shaft_diameter"] == pytest.approx(79.9)
    assert values["clearance"] == pytest.approx(50.0)
    assert kwargs["bore_diameter"] == pytest.approx(80e-3)
    assert kwargs["shaft_diameter"] == pytest.approx(79.9e-3)
    assert kwargs["clearance"] == pytest.approx(0.05e-3)


def test_form_field_order_matches_form_values_keys():
    values = bearing_to_form_values(CircularBearing())
    assert set(FORM_FIELD_ORDER) == set(values.keys())
