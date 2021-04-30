
# 👁‍🗨 Conspiracies

[![python versions](https://img.shields.io/badge/Python-%3E=3.7-blue)](https://github.com/KennethEnevoldsen/Conspiracies)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style.html)
[![release version](https://img.shields.io/badge/belief_graph%20Version-0.0.1-green)](https://github.com/KennethEnevoldsen/Conspiracies)
[![license](https://img.shields.io/github/license/KennethEnevoldsen/DaCy.svg?color=blue)](https://github.com/KennethEnevoldsen/Conspiracies)
[![github actions](https://github.com/KennethEnevoldsen/DaCy/actions/workflows/pytest.yml/badge.svg)](https://github.com/KennethEnevoldsen/Conspiracies/actions)
[![spacy](https://img.shields.io/badge/built%20with-spaCy-09a3d5.svg)](https://spacy.io)


## 🔧 Installation
The required packages are available in the `requirements.txt`, However, spacy-transformers should be installed from the fork by KennethEnevoldsen which adds the forward pass of attention (for more see [PR](https://github.com/explosion/spacy-transformers/pull/268)).

```
pip install -r requirements.txt 
pip install git+https://github.com/KennethEnevoldsen/spacy-transformers
pip install pydantic==1.8.1
```


## 🤤 Future improvements
Currently, the `lemmy` lemmatizer isn't used as it is incompatible with spacy version 3.

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


## License
Conspiracies is released under the Apache License, Version 2.0. See the `LICENSE` file for more details.

## Contact
Please use the [GitHub Issue Tracker](https://github.com/KennethEnevoldsen/conspiracies/issues) to contact us on this project.