# Quantifying reviewer recommendations with machine learning.

*Since eLife reviews do not contain an explicit review rating score, we perform sentiment analysis on the review text to quantify how positive it is. We quantify the reviewers’ recommendations to the editor for the submitted manuscript as follows. We first trained a classifier by fine-tuning a BERT deep neural language model (Devlin, et al. 2019) to predict review scores for a set of openly available reviews in a different field (machine learning), collected in the PeerRead corpus (Kang, et al., 2018). We then fine-tuned these reviews again, this time on a training set derived by manually scoring a subset of 900 eLife reviews on a 4-point scale of reject/mixed/accept/strong accept. Using a held-out test set, we found a high correlation (r =0.78) between the hand-scored reviews and the automatically scored reviews. Crucially, accuracy was 100% for reviews hand-scored at extreme values - i.e. there were no reviews hand-scored a 1 (“reject”) that were automatically scored above 3.0 or vice versa. The deep neural language model is implemented using PyTorch 1.4.0.*

*In addition, since there are multiple reviews (usually three) for the same submission, we control the submission quality by studying whether some reviewers consistently have higher or lower review ratings compared to other reviewers of the same submission. Following this intuition, we study the rank of each review’s rating among the reviews for the same submission. We normalize the rank by the number of reviews for the same submission. For example, for a submission with 3 reviews, the most negative reviewer gets a normalized rank of 0.0, the second most negative reviewer gets a normalized rank of 0.5, and the most positive reviewer gets a normalized rank of 1.0. We use the average review rating rank for studying the heterogeneity of reviewers*

## References

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019, June). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) (pp. 4171-4186).

Kang, D., Ammar, W., Dalvi, B., van Zuylen, M., Kohlmeier, S., Hovy, E., & Schwartz, R. (2018). A dataset of peer reviews (peerread): Collection, insights and nlp applications. arXiv preprint arXiv:1804.09635.



