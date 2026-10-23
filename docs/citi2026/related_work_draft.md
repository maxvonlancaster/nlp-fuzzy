# Related Work (Draft)

## Graph-based and algebraic extractive summarisation

Graph-based extractive methods treat sentences as nodes and their pairwise similarities as edges. TextRank applies the PageRank ranking model to such a graph to select sentences (Mihalcea and Tarau, 2004). LexRank similarly ranks sentences by their centrality in a lexical similarity graph, and is commonly applied with TF-IDF-based similarity (Erkan and Radev, 2004). Algebraic methods instead project sentences into a latent space. Latent semantic analysis (LSA) decomposes a term-sentence matrix with a singular value decomposition and selects sentences according to their weight in the leading latent topics (Steinberger and Ježek, 2004). These methods are unsupervised and interpretable, and they remain common baselines because they require no training data.

## Automatic evaluation and its limitations

ROUGE measures n-gram overlap between a candidate summary and reference summaries, and is the standard automatic metric for extractive and abstractive summarisation (Lin, 2004). Its rankings depend on the references used, and several studies have questioned whether ROUGE-based comparisons remain stable across evaluation settings. Bhandari et al. (2020) re-examine evaluation practice in text summarisation, and Mao et al. (2020) propose a facet-aware evaluation for extractive summarisation that looks beyond single-reference overlap. These works motivate examining how reference characteristics shape the outcome of a comparison, rather than relying on aggregate scores alone.

## Positional bias and the motivation for this study

Positional bias in summarisation has been studied mainly in news. Grenander et al. (2019) address lead bias in news summarisation, in which the opening sentences of an article carry much of the reference content, and propose multi-stage training with auxiliary losses to counter its effects. Their setting involves trained neural models on news corpora. Our study differs in three respects. It uses classical, training-free extractive methods; it operates on clinical transcriptions rather than news; and its references are the short `description` fields of MTSamples, which are proxy references and not verified gold-standard summaries. We do not introduce a new summarisation algorithm. Instead, we provide a dataset-specific empirical analysis that compares five unsupervised methods, measures the association between positional reference overlap and the advantage of Lead over LexRank, and checks whether that association persists after controlling for document and reference length.

## References

Bhandari, M., Gour, P. N., Ashfaq, A., Liu, P., and Neubig, G. (2020). Re-evaluating evaluation in text summarization. In *Proceedings of EMNLP 2020*. https://aclanthology.org/2020.emnlp-main.751/

Erkan, G., and Radev, D. R. (2004). LexRank: Graph-based lexical centrality as salience in text summarization. *Journal of Artificial Intelligence Research*, 22, 457–479. https://doi.org/10.1613/JAIR.1523

Grenander, M., Cao, Y., and Cohen, S. B. (2019). Countering the effects of lead bias in news summarization via multi-stage training and auxiliary losses. In *Proceedings of EMNLP-IJCNLP 2019*. https://aclanthology.org/D19-1620/

Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. In *Text Summarization Branches Out*. https://aclanthology.org/W04-1013/

Mao, Y., Liu, Y., Zhu, C., Yang, Z., Zhong, M., Han, J., and Hu, S. (2020). Facet-aware evaluation for extractive summarization. In *Proceedings of ACL 2020*. https://aclanthology.org/2020.acl-main.445/

Mihalcea, R., and Tarau, P. (2004). TextRank: Bringing order into text. In *Proceedings of EMNLP 2004*. https://aclanthology.org/W04-3252/

Steinberger, J., and Ježek, K. (2004). Using latent semantic analysis in text summarization and summary evaluation. In *Proceedings of ISIM 2004*. https://www.kiv.zcu.cz/~jstein/publikace/isim2004.pdf
