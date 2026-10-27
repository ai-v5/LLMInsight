# Third-party notices

LLMInsight is distributed under the AGPL-3.0-only license in `LICENSE`. The
project also includes or depends on the third-party components below. Their
licenses apply to those components and are not replaced by the project license.

## Bundled JavaScript

### Apache ECharts 5.5.1

- Component: `web/vendor/echarts.min.js`
- License: Apache License 2.0
- Upstream: <https://github.com/apache/echarts>
- License text: <https://www.apache.org/licenses/LICENSE-2.0>
- Bundled upstream license and subcomponent list: `licenses/echarts-5.5.1/LICENSE`
- Bundled upstream attribution notice: `licenses/echarts-5.5.1/NOTICE`
- Bundled d3 BSD 3-Clause license: `licenses/echarts-5.5.1/LICENSE-d3`

These files are copied verbatim from the ECharts 5.5.1 upstream tag, including
the d3 subcomponent identified in its LICENSE. The vendored JavaScript retains
its upstream header. All four notice/license files above, including this
document, are included in the wheel's `.dist-info/licenses/` directory.

## Python runtime dependencies

The versions installed by `requirements.txt` are selected by the package
resolver (`pandas>=2.0`, `numpy>=1.26`, and `pyyaml>=6.0`). Consult the exact
installed distribution metadata when producing a binary redistribution.

### pandas

- License: BSD 3-Clause License
- Project: <https://pandas.pydata.org/>
- License text: <https://github.com/pandas-dev/pandas/blob/main/LICENSE>

### NumPy

- License: BSD 3-Clause License
- Project: <https://numpy.org/>
- License text: <https://github.com/numpy/numpy/blob/main/LICENSE.txt>

### PyYAML

- License: MIT License
- Project: <https://pyyaml.org/>
- License text: <https://github.com/yaml/pyyaml/blob/main/LICENSE>

The Python standard library components used by LLMInsight are covered by the
Python Software Foundation License; they are not vendored by this repository.
