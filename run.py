# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 ams-OSRAM AG
# Copyright 2026 Maolin Wang (modifications)
#
# This repository redistributes and modifies Apache-2.0 licensed components.
# Upstream attributions: see NOTICE.

import argparse

from wimsoslam import config
from wimsoslam.system import WIMSOSystem

def main():
    parser = argparse.ArgumentParser(
        description="Run WIMSO-SLAM (dense RGB-D neural SLAM). / WIMSO-SLAM を実行します。"
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='Input dataset root (overrides the value in the YAML config). / 入力データのルート（YAMLより優先）')
    parser.add_argument('--output', type=str,
                        help='Output directory (overrides the value in the YAML config). / 出力先（YAMLより優先）')
    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configs/wimso_default.yaml')
    
    slam_sys = WIMSOSystem(cfg, args)

    slam_sys.run()

if __name__ == '__main__':
    main()
