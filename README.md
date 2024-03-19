## Variable Length Hidden Markov Models (VLHMM)

The repository concludes two modules *vlhmm*, *chipseq*, which can solve next problems:
* *vlhmm*:
    * building, training, plotting context transition tree
    * building, training VLHMM
* *chipseq*:
    * applying VLHMM to ChIp-seq data
    * conversion for genome browser fromat http://genome.ucsc.edu/

Language: Python 3.x

Dependencies:
* Cython
* NumPy, SciPy
* Pylab
* datrie
* PyGraphviz

Instalation:

    git clone https://github.com/atanna/hmm.git

Cython-files compilation:

    python setup.py build_ext --inplace
___
## Examples
The directory `vlhmm_/examples/` concludes test examples with training VLHMM and context trees on simulated data.

`chipseq/real_test.py` shows example VLHMM on ChIP-seq data
_ _ _
## Description
##### Context transition tree
* Context transition tree defines a stochastic process.
* Context state -- any preffix from previous states (the process moves from right to left, i.e. states go in descending order of time).
* Vertex ~ context.
* Edge ~ state.
* Outdegree of internal vertex -- number of states.
* The leaf defines the distribution of the current state.
* ###### Examples
    1. "Unfair Coin"<br>
![alt text](https://raw.githubusercontent.com/atanna/hmm/master/diploma/img/sample_mixture/real_trie_.png)

    2. Markov chain<br>
![alt text](https://raw.githubusercontent.com/atanna/hmm/master/diploma/img/sample_hmm1/real_trie_.png)

    3. Second-order Markov chain<br>
![alt text](https://raw.githubusercontent.com/atanna/hmm/master/diploma/img/Context_trie.png)
    4. Variable length Markov chain  
![alt text](https://raw.githubusercontent.com/atanna/hmm/master/diploma/img/Prune_c_trie.png)
*Context transition tree defines a variable leghth Markov stochastic process.

##### VLHMM
* The same as HMM.
* Hidden layer is defined by variable leghth Markov stochastic process (which we can define by context transition tree).

