# LBANN: Livermore Big Artificial Neural Network Toolkit

The Livermore Big Artificial Neural Network toolkit (LBANN) is an
open-source, HPC-centric, deep learning training framework that is
optimized to compose multiple levels of parallelism.

LBANN provides model-parallel acceleration through domain
decomposition to optimize for strong scaling of network training.  It
also allows for composition of model-parallelism with both data
parallelism and ensemble training methods for training large neural
networks with massive amounts of data.  LBANN is able to advantage of
tightly-coupled accelerators, low-latency high-bandwidth networking,
and high-bandwidth parallel file systems.

## DistConv Repository

The DistConv repository contains a a rewrite of the original DistConv
algorithm, published using the LBANN C++ Core, with a reimplmentation
using PyTorch 2.x DTensor objects.
 
## Publications

+ Yosuke Oyama, Naoya Maruyama, Nikoli Dryden, Erin McCarthy, Peter
  Harrington, Jan Balewski, Satoshi Matsuoka, Peter Nugent, Brian Van
  Essen. "The Case for Strong Scaling in Deep Learning: Training Large
  3D CNNs with Hybrid Parallelism", under review for Special Session
  on Parallel and Distributed Computing Techniques for AI, ML and DL
  in Transactions on Parallel and Distributed Systems, July 2020.

  + [arXiv.org/abs/2007.12856](https://arxiv.org/abs/2007.12856)

+ Nikoli Dryden, Naoya Maruyama, Tom Benson, Tim Moon, Marc Snir,
  Brian Van Essen. ["Channel and Filter Parallelism for Large-Scale
  CNN Training"](https://dl.acm.org/doi/10.1145/3295500.3356207), in
  *SC '19: Proceedings of the International Conference for High
  Performance Computing, Networking, Storage and Analysis*, November
  2019 Article No. 10, Pages 1-20, DOI:
  [10.1145/3295500.3356207](https://dl.acm.org/doi/10.1145/3295500.3356207).

```
    @INPROCEEDINGS{8820780,
      author={N. {Dryden} and N. {Maruyama} and T. {Benson} and T. {Moon} and M. {Snir} and B. {Van Essen}},
      booktitle={2019 IEEE International Parallel and Distributed Processing Symposium (IPDPS)},
      title={Improving Strong-Scaling of {CNN} Training by Exploiting Finer-Grained Parallelism},
      year={2019},
      volume={},
      number={},
      pages={210-220},
      doi={10.1109/IPDPS.2019.00031}}
```

+ Nikoli Dryden, Naoya Maruyama, Tom Benson, Tim Moon, Marc Snir,
  Brian Van Essen. ["Improving Strong-Scaling of CNN Training by
  Exploiting Finer-Grained
  Parallelism"](https://doi.org/10.1109/IPDPS.2019.00031), in
  [*Proceedings of IEEE International Parallel & Distributed
  Processing
  Symposium*](http://www.ipdps.org/ipdps2019/2019-advance-program.html),
  2019.

  + [arXiv.org/abs/1903.06681](https://arxiv.org/abs/1903.06681)
```
    @INPROCEEDINGS{8820780,
      author={N. {Dryden} and N. {Maruyama} and T. {Benson} and T. {Moon} and M. {Snir} and B. {Van Essen}},
      booktitle={2019 IEEE International Parallel and Distributed Processing Symposium (IPDPS)},
      title={Improving Strong-Scaling of {CNN} Training by Exploiting Finer-Grained Parallelism},
      year={2019},
      volume={},
      number={},
      pages={210-220},
      doi={10.1109/IPDPS.2019.00031}}
```

A complete list of LBANN related  publications, presentations and posters are shown
[here](https://lbann.readthedocs.io/en/latest/publications.html).

## Reporting issues
Issues, questions, and bugs can be raised on the [Github issue
tracker](https://github.com/LBANN/lbann/issues).
