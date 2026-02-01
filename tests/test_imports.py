# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 ams-OSRAM AG
# Copyright 2026 Maolin Wang (modifications)
#
# This repository redistributes and modifies Apache-2.0 licensed components.
# Upstream attributions: see NOTICE.

"""Lightweight sanity checks.

EN: These tests avoid importing heavy GPU dependencies in CI.
JP: CI では GPU 依存を避けるため、軽量チェックのみ行います。
"""

import pathlib


def test_package_layout():
    root = pathlib.Path(__file__).resolve().parents[1]
    assert (root / "wimsoslam").exists()
    assert (root / "wimsoslam" / "system.py").exists()
