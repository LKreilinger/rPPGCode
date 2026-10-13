# Third-party notices

The MIT License in [LICENSE](LICENSE) covers the code written for this project. It does **not** cover the files listed below. They come from other authors and stay under their own terms.

## PhysNet model and loss function (research use only)

Files:

- `cnn_process/load/PhysNet.py`
- `cnn_process/load/lossFunction.py`

Source: code of "Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks" by Zitong Yu, Xiaobai Li and Guoying Zhao (BMVC 2019), <https://github.com/ZitongYu/PhysNet>.

The original header of both files is kept unchanged. It states:

> Only for research purpose, and commercial use is not allowed.
> MIT License
> Copyright (c) 2019

Because of the first line, these two files may only be used for research and not commercially. The pipeline in this repository imports both files, so **the project as a whole can only be used for non-commercial research** unless these two files are replaced or their authors give permission.

If you use the code, the authors ask you to cite:

```bibtex
@inproceedings{yu2019remote,
    title={Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks},
    author={Yu, Zitong and Li, Xiaobai and Zhao, Guoying},
    booktitle= {British Machine Vision Conference (BMVC)},
    year = {2019}
}
```

Text of the MIT License the header refers to:

```text
MIT License

Copyright (c) 2019 Zitong Yu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Intel RealSense example

File: `data_collection/RealsenseGetFrames.py`

Based on the example `opencv_viewer_example.py` of the Intel RealSense SDK, <https://github.com/IntelRealSense/librealsense>, Copyright Intel Corporation, licensed under the Apache License, Version 2.0 (<https://www.apache.org/licenses/LICENSE-2.0>). The example was changed to save the frames to disk and to start and stop with the keyboard.

## PlotNeuralNet example

File: `data_collection/vis_model.py`

Based on an example of PlotNeuralNet by Haris Iqbal, <https://github.com/HarisIqbal88/PlotNeuralNet>, licensed under the MIT License.

## Datasets and Python packages

The datasets (UBFC-rPPG, PURE, UBFC-Phys) are not part of this repository. Each has its own terms of use, which you accept when you request the data from its authors.

The Python packages in the `requirements*.txt` files are installed separately and are not distributed with this repository. Each has its own license.
