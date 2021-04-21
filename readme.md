
# 👁‍🗨 Conspiracies

[![python versions](https://img.shields.io/badge/Python-%3E=3.7-blue)](https://github.com/KennethEnevoldsen/Conspiracies)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style.html)


## 🔧 Installation
The required packages are available in the `requirements.txt`, However, spacy-transformers should be installed from the fork by KennethEnevoldsen which adds the forward pass of attention (for more see [PR](https://github.com/explosion/spacy-transformers/pull/268)).

```
pip install -r requirements.txt 
pip install git+https://github.com/KennethEnevoldsen/spacy-transformers
```

To run the Danish language pipeline you will also need:
```
python -m spacy download da_core_news_sm
```

<!-- 
### References

If you use this library in your research, please kindly cite:

```bibtex
@inproceedings{enevoldsen2020dacy,
    title={DaCy: A SpaCy NLP Pipeline for Danish},
    author={Enevoldsen, Kenneth},
    year={2021}
}
```
-->
