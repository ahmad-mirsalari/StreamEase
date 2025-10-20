
# StreamEase
A python library that automates the conversion of Temporal Convolutional Network (TCN) models to a streaming format without affecting model accuracy, significantly reducing deployment time and computational
resources.

## References
This library is based on the research outlined in the following papers:

 - S. A. Mirsalari, L. Bijar, M. Fariselli, M. Croome, F. Paci, G. Tagliavini, L. Benini, "StreamEase: Enabling Real-Time Inference of Temporal Convolution Networks on Low-Power MCUs with Stream-Oriented Automatic Transformation", 2024 31st IEEE International Conference on Electronics, Circuits, and Systems (ICECS), Nancy, France, 2024, pp. 1-4, [Link to the paper](https://ieeexplore.ieee.org/abstract/document/10848742)
- S. A. Mirsalari, M. Fariselli, L. Bijar, F. Paci, L. Benini and G. Tagliavini, "Enabling Real-Time Streaming Temporal Convolution Network Inference on Ultra-Low-Power Microcontrollers," 2025 IEEE Computer Society Annual Symposium on VLSI (ISVLSI), Kalamata, Greece, 2025, pp. 1-6, doi: 10.1109/ISVLSI65124.2025.11130291. [Link to the paper](https://ieeexplore.ieee.org/abstract/document/11130291)
  
If you find this library useful in your research, please consider citing the papers:

 > ```
> @INPROCEEDINGS{10848742,
> author={Mirsalari, Seyed Ahmad and Bijar, Léo and Fariselli, Marco and Croome, Martin and Paci, Francesco and Tagliavini, Giuseppe and Benini, Luca},
>  booktitle={2024 31st IEEE International Conference on Electronics, Circuits and Systems (ICECS)}, 
>  title={StreamEase: Enabling Real-Time Inference of Temporal Convolution Networks on Low-Power MCUs with Stream-Oriented Automatic Transformation}, 
>  year={2024},
>  volume={},
>  number={},
>  pages={1-4},
>  keywords={Accuracy;Computational modeling;Time series analysis;Memory management;Green products;Real-time systems;Libraries;Hardware;Integrated circuit modeling;Faces;streaming TCN;parallel ultra-low-power plat-form;real-time inference},
>  doi={10.1109/ICECS61496.2024.10848742}}
> ```

> ```
> @INPROCEEDINGS{11130291,
> author={Mirsalari, Seyed Ahmad and Fariselli, Marco and Bijar, Léo and Paci, Francesco and Benini, Luca and Tagliavini, Giuseppe},
> booktitle={2025 IEEE Computer Society Annual Symposium on VLSI (ISVLSI)}, 
> title={Enabling Real-Time Streaming Temporal Convolution Network Inference on Ultra-Low-Power Microcontrollers}, 
> year={2025},
> volume={1},
> number={},
> pages={1-6},
> keywords={Performance evaluation;Adaptation models;Quantization (signal);Microcontrollers;Computational modeling;Speech enhancement;Very large scale integration;Real-time systems;Computational efficiency;Optimization;TCN;Streaming Applications;MCU;Embedded systems},
> doi={10.1109/ISVLSI65124.2025.11130291}}
> ```
 
## Setup
#### Clone the Repository:
~~~~~shell
git clone git@github.com:ahmad-mirsalari/StreamEase.git
cd StreamEase
~~~~~

#### Install Dependencies:
<!-- To ensure compatibility when checking network functionality, particularly with onnxruntime, I recommend using Python 3.10. This version has been found to work well with onnxruntime, avoiding some version conflicts that might occur with newer Python versions. -->
You can install the required dependencies by running:
~~~~~shell
pip install -r requirements.txt
~~~~~
#### Set Up the tool:
Run the setup_env.sh script to add the project root to your PYTHONPATH. This will ensure that Python can locate the streamease package.
~~~~~shell
chmod +x setup_env.sh
./setup_env.sh
~~~~~
You will be prompted to add the PYTHONPATH to your .bashrc to make it permanent for future sessions.

## Repository Organization
- [streamease](./streamease/): Main package containing modules for streaming inference, buffering, and quantization.
- [examples](./examples/): Contains example scripts demonstrating how to use the toolkit.
- [utils](./utils/): Contains helper functions to facilitate running the streaming network, with support for ONNX Runtime and GreenWaves Technologies' nntool for optimized model execution.
- [setup_env.sh](setup_env.sh): Script for setting up dependencies and environment variables.
- [requirements.txt](requirements.txt): List of required dependencies

## Roadmap

- Extend the library by adding additional networks in different applications


## License 
 StreamEase is released under Apache 2.0, see the [LICENSE](./LICENSE.md) file in the root of this repository for details.

## Acknowledgements
This work was supported by the [APROPOS](https://projects.tuni.fi/apropos/) project (g.a. no. 956090), founded by the European Union’s Horizon 2020 research and innovation program. 


## Contributors
- [Seyed Ahmad Mirsalari](https://github.com/ahmad-mirsalari), University of Bologna,[E-mail](mailto:seyedahmad.mirsalar2@unibo.it)


## 🚀 Contact Me
- [Email](mailto:seyedahmad.mirsalar2@unibo.it)
- [LinkedIn](https://www.linkedin.com/in/ahmad-mirsalari/)
- [Twitter](https://twitter.com/ahmad_mirsalari)
- [APROPOS](https://projects.tuni.fi/apropos/news/pr_esr_3/)
