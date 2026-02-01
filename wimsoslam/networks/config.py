# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 ams-OSRAM AG
# Copyright 2026 Maolin Wang (modifications)
#
# This repository redistributes and modifies Apache-2.0 licensed components.
# Upstream attributions: see NOTICE.

from wimsoslam.networks.decoders import Decoders

def get_model(cfg):
    c_dim = cfg['model']['c_dim']  # feature dimensions
    truncation = cfg['model']['truncation']
    learnable_beta = cfg['rendering']['learnable_beta']

    decoder = Decoders(c_dim=c_dim, truncation=truncation, learnable_beta=learnable_beta)
    # decoder = Decoders(c_dim=c_dim, conf_dim=50, col_conf_dim=75, truncation=truncation, learnable_beta=learnable_beta)

    return decoder
