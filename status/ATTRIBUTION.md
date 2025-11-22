# YiRage - Attribution and Acknowledgments

## Derivative Work

**YiRage** (Yield Revolutionary AGile Engine)  
- Developer: Chen Xingqiang (陈星强)
- Copyright: 2025
- License: Apache License 2.0
- Repository: https://github.com/chenxingqiang/yirage

## Original Work

**Mirage** - A Multi-Level Superoptimizer for Tensor Programs  
- Developed at: Carnegie Mellon University
- Copyright: 2023-2024 CMU
- License: Apache License 2.0
- Repository: https://github.com/mirage-project/mirage
- Paper: [Mirage: A Multi-Level Superoptimizer for Tensor Programs](https://arxiv.org/abs/2405.05751)

## YiRage Enhancements

YiRage extends Mirage with comprehensive multi-backend support:

### Multi-Backend Architecture
- **7 Complete Backend Implementations**
  - CUDA (NVIDIA GPU)
  - CPU (x86/ARM processors)
  - MPS (Apple Silicon)
  - Triton (Compiler backend)
  - NKI (AWS Neuron)
  - cuDNN (CUDA acceleration)
  - MKL (Intel acceleration)

### Hardware-Aware Optimizations
- **42+ Optimization Methods** across 7 backends
- **CUDA**: Tensor Core, Warp scheduling, Bank conflict avoidance
- **CPU**: SIMD detection (AVX512/AVX2/AVX/SSE), Cache blocking, OpenMP
- **MPS**: GPU family detection (M1/M2/M3), Threadgroup optimization
- **Triton**: Block configuration, Software pipelining
- **NKI**: SBUF optimization, DMA scheduling, NeuronCore tiling
- **cuDNN**: Algorithm selection, Math type configuration
- **MKL**: Threading modes, BLAS integration

### Search Strategies
- **5 Independent Search Strategies** with hardware-specific optimization
- **15 Candidate Generation Dimensions**
- **13 Performance Evaluation Metrics**
- Auto-tuning and performance modeling

### Code Contributions
- **67 New Files** (16,900+ lines)
  - 35 C++ header files
  - 25 C++ source files
  - 1 Python module
  - 3 build configuration files
  - 11 documentation files
  - 2 tests and examples
  - 1 validation script

### Documentation
- Complete design documentation
- Comprehensive usage guides
- Implementation reports
- Performance optimization guides

## Acknowledgments

We thank the Mirage team at Carnegie Mellon University for creating the foundational
superoptimizer framework. YiRage builds upon Mirage's excellent architecture and
extends it with multi-backend capabilities.

Special thanks to:
- The original Mirage development team at CMU
- Prof. Zhihao Jia and collaborators
- The open-source community

## License

Both Mirage and YiRage are licensed under the Apache License 2.0, which permits
derivative works under the same license.

See [LICENSE](LICENSE) file for details.

## Citation

If you use YiRage in your research, please cite both:

**YiRage**:
```bibtex
@software{yirage2025,
  title={YiRage: Yield Revolutionary AGile Engine for Multi-Backend LLM Inference},
  author={Chen, Xingqiang},
  year={2025},
  note={A derivative work based on Mirage},
  url={https://github.com/chenxingqiang/yirage}
}
```

**Original Mirage**:
```bibtex
@inproceedings{wu2024mirage,
  title={Mirage: A Multi-Level Superoptimizer for Tensor Programs}, 
  author={Mengdi Wu and Xinhao Cheng and Shengyu Liu and Chunan Shi and Jianan Ji and Kit Ao and Praveen Velliengiri and Xupeng Miao and Oded Padon and Zhihao Jia},
  booktitle={19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 25)},
  year={2025},
  address={Boston, MA},
  publisher={USENIX Association},
  month=jul
}
```

---

**Last Updated**: 2025-11-21  
**YiRage Version**: 1.0.0  
**Based on**: Mirage (CMU)

