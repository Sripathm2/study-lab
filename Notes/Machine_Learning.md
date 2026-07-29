# ML Notes

A running file of machine-learning concepts, in Definition / Intuition / Notes form. Written to stand alone — no book required to read them. Math is kept light: a one-line formula plus plain English, not derivations.

Statistics foundations — statistical-learning theory, linear regression, classification, resampling, model selection & regularization, moving beyond linearity, survival analysis, and multiple testing — live in a companion **Stat.md**. Cross-references to those concepts are marked *(stat)*.

Sources here: *An Introduction to Statistical Learning* (ISL/ISLP), *Hands-On Machine Learning* (Géron), Karpathy's *Zero to Hero*, Chip Huyen's *Designing Machine Learning Systems*, and Kohavi et al.'s *Trustworthy Online Controlled Experiments*. Section tags show which pass a topic came from.

---

## Contents

**Data Preparation** *(Hands-On ML)*
- [Feature scaling](#feature-scaling)

**Tree-Based Methods** *(ISL ch. 8)*
- [Decision trees](#decision-trees)
- [Regression trees](#regression-trees)
- [Tree pruning (cost-complexity)](#tree-pruning-cost-complexity)
- [Classification trees](#classification-trees)
- [Trees: strengths and weaknesses](#trees-strengths-and-weaknesses)
- [Ensemble methods and weak learners](#ensemble-methods-and-weak-learners)
- [Bagging](#bagging)
- [Out-of-bag (OOB) error](#out-of-bag-oob-error)
- [Random forests](#random-forests)
- [Boosting](#boosting)
- [Bayesian additive regression trees (BART)](#bayesian-additive-regression-trees-bart)
- [Gradient boosting libraries (XGBoost, LightGBM, CatBoost)](#gradient-boosting-libraries-xgboost-lightgbm-catboost)

**Support Vector Machines** *(ISL ch. 9)*
- [Hyperplane](#hyperplane)
- [Maximal margin classifier](#maximal-margin-classifier)
- [Support vector classifier (soft margin)](#support-vector-classifier-soft-margin)
- [Support vector machine (kernels)](#support-vector-machine-kernels)
- [Mercer's theorem](#mercers-theorem)
- [Hinge loss](#hinge-loss)
- [SVMs with more than two classes](#svms-with-more-than-two-classes)

**Neural Networks and Deep Learning** *(ISL ch. 10; Hands-On ML; Karpathy)*
- [Artificial neuron and the perceptron](#artificial-neuron-and-the-perceptron)
- [Dense (fully connected) layer](#dense-fully-connected-layer)
- [Perceptron learning rule](#perceptron-learning-rule)
- [Multi-layer perceptron (MLP) and the XOR problem](#multi-layer-perceptron-mlp-and-the-xor-problem)
- [Backpropagation](#backpropagation)
- [Computational graph and autograd](#computational-graph-and-autograd)
- [Zeroing gradients](#zeroing-gradients)
- [Numerical stability](#numerical-stability)
- [Weight initialization](#weight-initialization)
- [Activation saturation](#activation-saturation)
- [Batch normalization](#batch-normalization)
- [WaveNet / hierarchical context](#wavenet--hierarchical-context)
- [Train / dev / test split](#train--dev--test-split)
- [Neural networks (feed-forward)](#neural-networks-feed-forward)
- [Hidden layers and units](#hidden-layers-and-units)
- [Activation function](#activation-function)
- [Output layer and loss](#output-layer-and-loss)
- [Softmax](#softmax)
- [Logits](#logits)
- [One-hot encoding](#one-hot-encoding)
- [Likelihood and negative log-likelihood](#likelihood-and-negative-log-likelihood)
- [Language model](#language-model)
- [Broadcasting](#broadcasting)
- [Convolutional neural network (CNN)](#convolutional-neural-network-cnn)
- [Convolution filter](#convolution-filter)
- [Pooling](#pooling)
- [Data augmentation](#data-augmentation)
- [Bag-of-words](#bag-of-words)
- [Recurrent neural network (RNN)](#recurrent-neural-network-rnn)
- [Embedding layer](#embedding-layer)
- [LSTM](#lstm)
- [Autoregressive models and autocorrelation](#autoregressive-models-and-autocorrelation)
- [Gradient descent](#gradient-descent)
- [Stochastic gradient descent (SGD)](#stochastic-gradient-descent-sgd)
- [Learning-rate schedule](#learning-rate-schedule)
- [Adam and AdamW](#adam-and-adamw)
- [Dropout](#dropout)
- [Neural network regularization](#neural-network-regularization)
- [Max-norm regularization](#max-norm-regularization)
- [MC dropout](#mc-dropout)
- [Early stopping](#early-stopping)
- [Vanishing and exploding gradients](#vanishing-and-exploding-gradients)
- [Double descent](#double-descent)

**Transformers and Large Language Models** *(Karpathy)*
- [Tokenization and byte-pair encoding (BPE)](#tokenization-and-byte-pair-encoding-bpe)
- [Regex pre-tokenization](#regex-pre-tokenization)
- [Special tokens](#special-tokens)
- [tiktoken vs SentencePiece](#tiktoken-vs-sentencepiece)
- [Tokenization failure modes](#tokenization-failure-modes)
- [Adding vocabulary after training](#adding-vocabulary-after-training)
- [Self-attention](#self-attention)
- [Scaled dot-product attention](#scaled-dot-product-attention)
- [Causal (masked) self-attention](#causal-masked-self-attention)
- [Multi-head attention](#multi-head-attention)
- [Positional encoding](#positional-encoding)
- [Residual (skip) connections](#residual-skip-connections)
- [Layer normalization](#layer-normalization)
- [Transformer feed-forward network](#transformer-feed-forward-network)
- [Transformer block](#transformer-block)
- [Encoder, decoder, and cross-attention](#encoder-decoder-and-cross-attention)
- [Decoder-only transformer (GPT)](#decoder-only-transformer-gpt)
- [GPT training pipeline](#gpt-training-pipeline)
- [Pretraining and base models](#pretraining-and-base-models)
- [Supervised fine-tuning (SFT)](#supervised-fine-tuning-sft)
- [Reward modeling and RLHF](#reward-modeling-and-rlhf)
- [Mode collapse](#mode-collapse)
- [In-context learning (zero/one/few-shot)](#in-context-learning-zeroonefew-shot)
- [Chain-of-thought prompting](#chain-of-thought-prompting)
- [Self-consistency and ensembling attempts](#self-consistency-and-ensembling-attempts)
- [Tree of thoughts](#tree-of-thoughts)
- [LLM agents (chains, ReAct)](#llm-agents-chains-react)
- [Retrieval-augmented generation (RAG)](#retrieval-augmented-generation-rag)
- [Constrained prompting](#constrained-prompting)
- [Parameter-efficient fine-tuning (LoRA / PEFT)](#parameter-efficient-fine-tuning-lora--peft)
- [LLM limitations and safe use](#llm-limitations-and-safe-use)
- [torch.compile](#torchcompile)

**Unsupervised Learning** *(ISL ch. 12)*
- [Unsupervised learning](#unsupervised-learning)
- [Principal components analysis (PCA)](#principal-components-analysis-pca)
- [Matrix completion](#matrix-completion)
- [Clustering](#clustering)
- [K-means clustering](#k-means-clustering)
- [Hierarchical clustering](#hierarchical-clustering)
- [Gaussian mixture model (GMM)](#gaussian-mixture-model-gmm)
- [Bayesian Gaussian mixture](#bayesian-gaussian-mixture)

**Machine Learning Systems Design** *(Designing ML Systems)*
- [When to use machine learning](#when-to-use-machine-learning)
- [Classification task types (binary, multiclass, multilabel)](#classification-task-types-binary-multiclass-multilabel)
- [Decoupling objectives (multi-objective optimization)](#decoupling-objectives-multi-objective-optimization)
- [ETL (extract, transform, load)](#etl-extract-transform-load)
- [Non-probability sampling](#non-probability-sampling)
- [Random sampling methods (simple, stratified, weighted, reservoir)](#random-sampling-methods-simple-stratified-weighted-reservoir)
- [Labeling and weak supervision](#labeling-and-weak-supervision)
- [Transfer learning](#transfer-learning)
- [Zero-shot learning](#zero-shot-learning)
- [Active learning](#active-learning)
- [Class imbalance](#class-imbalance)
- [Handling missing data](#handling-missing-data)
- [Categorical feature encoding (hashing trick)](#categorical-feature-encoding-hashing-trick)
- [Feature crossing](#feature-crossing)
- [Data leakage](#data-leakage)
- [Classification metrics (precision, recall, F1)](#classification-metrics-precision-recall-f1)
- [Stacking (stacked generalization)](#stacking-stacked-generalization)
- [Distributed training (data vs model parallelism)](#distributed-training-data-vs-model-parallelism)
- [Model evaluation tests (perturbation, invariance, slice-based)](#model-evaluation-tests-perturbation-invariance-slice-based)
- [Batch features vs streaming features](#batch-features-vs-streaming-features)
- [Batch prediction vs online prediction](#batch-prediction-vs-online-prediction)
- [Model compression (distillation, pruning, quantization)](#model-compression-distillation-pruning-quantization)
- [Data distribution shifts (covariate, label, concept)](#data-distribution-shifts-covariate-label-concept)
- [Degenerate feedback loops](#degenerate-feedback-loops)
- [Continual learning (stateless vs stateful)](#continual-learning-stateless-vs-stateful)
- [Deployment strategies (shadow, canary, A/B testing)](#deployment-strategies-shadow-canary-ab-testing)
- [Bandit algorithms](#bandit-algorithms)
- [Storage vs compute](#storage-vs-compute)

**Controlled Experiments and A/B Testing** *(Trustworthy Online Controlled Experiments)*
- [A/B testing (controlled experiments)](#ab-testing-controlled-experiments)
- [Overall Evaluation Criterion (OEC)](#overall-evaluation-criterion-oec)
- [Parameters, variants, and randomization unit](#parameters-variants-and-randomization-unit)
- [Metric taxonomy (goal, driver, guardrail)](#metric-taxonomy-goal-driver-guardrail)
- [Good metric properties (measurable, attributable, sensitive)](#good-metric-properties-measurable-attributable-sensitive)
- [Twyman's law and trustworthiness](#twymans-law-and-trustworthiness)

**[Glossary](#glossary)** — alphabetical index

---

## Data Preparation *(Hands-On ML)*

### Feature scaling

**Definition.** Putting all attributes on the same scale, by one of two common routes. *Min-max scaling* (often called normalization): subtract the minimum and divide by the range (max − min), landing values in `[0, 1]`. *Standardization*: subtract the mean and divide by the standard deviation, giving mean 0 and unit variance.

**Intuition.** Algorithms that measure distances or penalize coefficient size treat a variable measured in thousands as inherently "bigger" than one measured in units — scaling removes that accident of measurement. Min-max gives a bounded range; standardization doesn't bound the output but is far less disturbed by outliers, since one extreme value doesn't squash everything else into a sliver of `[0, 1]`.

**Notes.** Already required by several methods in this file: ridge and the lasso (the penalty scales with coefficient size), PCA (sensitive to units), KNN and radial-kernel SVMs (distance-based), hierarchical clustering, and neural networks. Scikit-learn provides `MinMaxScaler` (with a `feature_range` option) and `StandardScaler`. Fit the scaler on the training set only, then apply it to the test set. → Ridge regression (stat), Principal components analysis (PCA), K-nearest neighbors (KNN) (stat), Hierarchical clustering.

---

## Tree-Based Methods *(ISL ch. 8)*

### Decision trees

**Definition.** Methods that split the predictor space into simple, non-overlapping regions and make a constant prediction in each. Drawn upside down: split points are *internal nodes*, the final regions are *terminal nodes* (leaves).

**Intuition.** A flowchart of yes/no questions on the predictors; follow the branches to a leaf, and the leaf gives your prediction. Splits carve the feature space into boxes.

**Notes.** Come in regression and classification flavors. → Regression trees, Classification trees, Trees: strengths and weaknesses.

### Regression trees

**Definition.** Divide the predictor space into `J` boxes `R_1, …, R_J`; predict, for any point in `R_j`, the *mean* response of the training observations in `R_j`. Boxes are found by *recursive binary splitting* — a top-down, greedy procedure that at each step picks the predictor and cutpoint giving the greatest immediate reduction in RSS.

**Intuition.** Repeatedly ask "which single split most reduces error right now?" and take it, never looking ahead. Greedy because it optimizes the current step, not the whole tree; top-down because it starts with all data in one region and keeps splitting.

**Notes.** A fully grown tree overfits (too many boxes → low bias, high variance). The fix is to grow big then prune. → Tree pruning.

### Tree pruning (cost-complexity)

**Definition.** Grow a large tree, then trim it back to a subtree using *cost-complexity (weakest-link) pruning*: minimize `RSS + α·|T|`, where `|T|` is the number of leaves and `α ≥ 0` is a tuning parameter. Choose `α` by cross-validation.

**Intuition.** A big tree fits noise; a smaller one generalizes better at the cost of a little bias. The `α·|T|` term charges rent per leaf, so raising `α` prunes branches in a predictable nested order. Cross-validation finds the `α` (tree size) that minimizes test error.

**Notes.** Cheaper than evaluating every possible subtree — increasing `α` sweeps out the whole useful sequence of subtrees automatically. → Regression trees, k-fold cross-validation (stat).

### Classification trees

**Definition.** Trees for a qualitative response: predict the *most common class* in each leaf. Splits are chosen to increase node *purity*, measured by the *Gini index* or *entropy* rather than RSS.

**Intuition.** Same recursive splitting, but "good split" now means "makes the resulting groups more class-pure." Gini and entropy both reward leaves dominated by a single class.

**Notes.** Purity measures are more sensitive than raw error rate for growing the tree. → Regression trees, Decision trees.

### Trees: strengths and weaknesses

**Definition.** Single trees are highly interpretable but relatively weak predictors.

**Intuition.** *Strengths:* easy to explain (even easier than linear regression), arguably mirror human decision-making, display graphically, and handle qualitative predictors without dummy variables. *Weaknesses:* lower predictive accuracy than top methods, and *non-robust* — a small data change can swing the whole tree.

**Notes.** Aggregating many trees — bagging, random forests, boosting, BART — trades away interpretability for big accuracy gains, largely by cutting the high variance of single trees. → Bagging, Random forests, Boosting.

### Ensemble methods and weak learners

**Definition.** An *ensemble* combines many simple "building-block" models (*weak learners*, individually mediocre) into one much stronger model.

**Intuition.** Many weak, uncorrelated opinions, aggregated, beat any single one — a crowd-of-experts effect. Trees make ideal building blocks. Bagging, random forests, boosting, and BART are all tree ensembles.

**Notes.** They differ in *how* they build and combine the trees. → Bagging, Random forests, Boosting, BART.

### Bagging

**Definition.** *Bootstrap aggregation*: draw `B` bootstrap samples, fit a tree on each, and average their predictions (regression) or take a majority vote (classification). A general variance-reduction procedure, especially useful for trees.

**Intuition.** Averaging many noisy estimates cuts variance (`Var` of an average of `n` items is `σ²/n`). Single trees are high-variance; average many bootstrap-grown trees and the wobble largely cancels, boosting accuracy. Trees are grown deep and *unpruned* — bagging handles the resulting variance.

**Notes.** More trees `B` never causes overfitting (it just stops helping), so pick `B` large enough to level off. Cost: you lose the single-tree picture, so interpretability drops; recover some via variable-importance measures (total RSS or Gini reduction per predictor). Test error comes free via OOB. → Out-of-bag error, Random forests.

### Out-of-bag (OOB) error

**Definition.** Each bagged tree is trained on a bootstrap sample that omits about one-third of observations (its *out-of-bag* points). Predict each observation using only the trees where it was OOB, and average — yielding a test-error estimate without cross-validation.

**Intuition.** Every point is naturally a hold-out for the roughly one-third of trees that never saw it. Grading each point on exactly those trees gives an honest, essentially free validation.

**Notes.** Convenient for large bagged models where cross-validation would be expensive. → Bagging.

### Random forests

**Definition.** Bagging with a twist that *decorrelates* the trees: at each split, only a random subset of `m` predictors (out of `p`) is allowed as split candidates.

**Intuition.** In plain bagging, if one predictor is very strong, nearly every tree splits on it first, so the trees look alike and averaging them barely helps. Forcing each split to ignore most predictors lets other variables lead sometimes, making the trees different — and averaging *different* trees cuts variance far more.

**Notes.** `m` is the key knob: `m = p` is just bagging; small `m` helps most when predictors are many and correlated (a common default is `m ≈ √p`). → Bagging, Variance (stat).

### Boosting

**Definition.** Grows trees *sequentially*, each one fit to the *residuals* (the signal left over) from the current model, then added in shrunken. No bootstrap sampling — each tree uses a modified version of the original data. Three tuning parameters: number of trees `B`, shrinkage `λ` (learning rate), and tree depth `d` (interaction depth).

**Intuition.** Learn slowly and correct your own mistakes: fit a small tree, see what it still gets wrong, fit the next tree to *those* errors, and repeat, nudging the model a little each time. Small trees (even `d = 1` stumps) added gradually build a strong additive model.

**Notes.** Unlike bagging/forests, boosting *can* overfit if `B` is too large (slowly) — choose `B` by cross-validation. Small `λ` (0.01, 0.001) learns more carefully but needs larger `B`. `d` controls how many predictors can interact. → Bagging, Random forests, BART.

### Bayesian additive regression trees (BART)

**Definition.** A tree ensemble that blends the ideas of forests and boosting: trees are grown with randomness (like forests) *and* successively try to capture signal the current model misses (like boosting), by *perturbing* each tree from the previous iteration. Predictions are averaged over many iterations after a burn-in.

**Intuition.** Each round, randomly tweak the existing trees, keeping changes that improve the fit — a guided random search that both chases leftover signal and avoids getting stuck in one solution, exploring the space of models thoroughly.

**Notes.** Places among the tree ensembles by how it builds trees: bagging (independent, on resamples), random forests (independent, random feature subsets), boosting (sequential, no resampling, slow learning), BART (sequential, perturbation-based, avoids local minima). → Ensemble methods, Boosting.

### Gradient boosting libraries (XGBoost, LightGBM, CatBoost)

**Definition.** Production implementations of gradient-boosted trees. *XGBoost* and *LightGBM* are heavily optimized, regularized boosters (LightGBM grows trees leaf-wise and is very fast on large data). *CatBoost* adds two distinctive ideas: **native categorical handling** (ordered target statistics instead of manual one-hot) and **ordered boosting** (a permutation scheme that prevents the target leakage ordinary boosting suffers from).

**Intuition.** Plain boosting is the *algorithm*; these are the batteries-included *libraries* that make it fast, regularized, and practical. CatBoost's ordered boosting computes each row's residual from a model trained only on *earlier* rows, so a row's own label never leaks into its encoding or gradient — cutting overfitting, especially with categorical features.

**Example.** On a table with a high-cardinality `city` column, CatBoost encodes each city by an *ordered* target statistic (the average label over previously-seen rows) instead of exploding it into thousands of one-hot columns — often better accuracy with far less preprocessing.

**Notes.** All three routinely top tabular-data benchmarks; the pick is usually empirical. *(beyond ISL)* → Boosting, Categorical feature encoding (hashing trick), Bagging.

---

## Support Vector Machines *(ISL ch. 9)*

### Hyperplane

**Definition.** In `p`-dimensional space, a hyperplane is a flat `(p − 1)`-dimensional subspace — a line in 2D, a plane in 3D. A *separating hyperplane* perfectly divides two labeled classes, with `β_0 + β_1 x_1 + … + β_p x_p > 0` on one side and `< 0` on the other (equivalently `y_i·(that expression) > 0` for all `i`).

**Intuition.** A flat divider one dimension below the space it lives in. If two classes can be split by such a divider, the sign of the linear score tells you which side — and which class — a point is on.

**Notes.** When a separating hyperplane exists, infinitely many do (nudge or tilt it slightly). Picking the "best" one motivates the maximal margin classifier. → Maximal margin classifier.

### Maximal margin classifier

**Definition.** Among all separating hyperplanes, the one *farthest* from the training points. The *margin* is the smallest distance from any point to the hyperplane; the maximal margin (optimal separating) hyperplane makes that smallest distance as large as possible. The points touching the margin are the *support vectors*.

**Intuition.** Split the classes with the widest possible buffer zone, giving the most breathing room on both sides. Only the closest points — the support vectors — define the boundary; the rest don't matter.

**Notes.** Fatal limitation: it *requires* the classes to be linearly separable, and even then it's hypersensitive to individual points (the boundary is fixed by a few support vectors). The support vector classifier relaxes both problems. → Support vector classifier.

### Support vector classifier (soft margin)

**Definition.** A *soft-margin* generalization of the maximal margin classifier: allow some points to sit inside the margin, or even on the wrong side of the hyperplane, controlled by a budget/tuning parameter `C`.

**Intuition.** Insisting on perfect separation makes the boundary brittle and often impossible. Deliberately tolerating a few violations buys a boundary that classifies *most* points more robustly. `C` sets how many/how severe the violations allowed — a bias–variance dial (loose `C` = wider margin, more bias, less variance).

**Notes.** Only points on or violating the margin (the support vectors) affect the boundary, making it robust to distant points. Still linear, though. → Maximal margin classifier, Support vector machine, Bias–variance trade-off (stat).

### Support vector machine (kernels)

**Definition.** Extends the support vector classifier to *non-linear* boundaries by enlarging the feature space through *kernels* — functions that act as inner products in a higher-dimensional space without computing the coordinates explicitly. Common choices: the *polynomial kernel* and the *radial (RBF) kernel*.

**Intuition.** A straight divider in a cleverly expanded feature space is a curved divider in the original space. Kernels let you get that curvature efficiently, without ever building the huge expanded features. The radial kernel is *local*: only training points near a test point influence its label (distant points contribute almost nothing).

**Intuition (the kernel trick).** You could add the extra features by hand — polynomial features, or *similarity features* measuring how close each instance is to chosen landmarks. Both work with any learning algorithm, but computing all those extra features is expensive on large training sets. The kernel trick gets you the same result *as if* you had added them, without ever adding them. *(Hands-On ML)*

**Notes.** Often one of the best "out-of-the-box" classifiers. Kernel choice and its parameters (e.g. `γ` for radial, degree for polynomial) tune flexibility. → Support vector classifier, Mercer's theorem, Hinge loss, SVMs with more than two classes.

### Mercer's theorem

**Definition.** If a function `K(a, b)` satisfies *Mercer's conditions* (continuous, symmetric so `K(a, b) = K(b, a)`, and a few others), then there exists a mapping `φ` into some other — possibly much higher-dimensional — space with `K(a, b) = φ(a)ᵀ φ(b)`.

**Intuition.** It's the licence for the kernel trick: you can use `K` as a kernel knowing a feature map `φ` exists, even though you never learn what `φ` is or compute it. For the Gaussian RBF kernel, `φ` maps each instance into an *infinite*-dimensional space — a strong argument for not doing the mapping explicitly.

**Notes.** Some kernels used in practice (e.g. the sigmoid kernel) don't satisfy all of Mercer's conditions yet generally work well anyway. *(Hands-On ML)* → Support vector machine (kernels).

### Hinge loss

**Definition.** The loss function underlying SVM training: `max(0, 1 − t)`. It equals 0 once `t ≥ 1`; its slope is `−1` for `t < 1` and `0` for `t > 1`.

**Intuition.** Zero penalty once an instance is correctly classified with enough margin, and a penalty growing linearly as it falls short — exactly the "get on the right side, with room to spare" objective of a margin classifier.

**Notes.** Not differentiable at `t = 1`, but (as with the lasso's absolute-value penalty) gradient-based optimization still works using a subgradient. *(Hands-On ML)* → Support vector classifier, The lasso (stat).

### SVMs with more than two classes

**Definition.** SVMs are inherently two-class; two schemes extend them to `K > 2`. *One-versus-one*: build an SVM for every pair of classes and let them vote. *One-versus-all (one-versus-rest)*: build `K` SVMs, each separating one class from the rest, and assign the class most confidently positive.

**Intuition.** Break a many-class problem into many two-class problems, then aggregate. One-vs-one runs more classifiers but on cleaner pairwise splits; one-vs-all runs fewer but on harder one-against-everything splits.

**Notes.** The separating-hyperplane idea doesn't generalize cleanly beyond two classes, which is why these workarounds exist. → Support vector machine.

---

## Neural Networks and Deep Learning *(ISL ch. 10; Hands-On ML; Karpathy)*

### Artificial neuron and the perceptron

**Definition.** The original *artificial neuron* takes one or more binary on/off inputs and fires when enough of its inputs are active. The *perceptron* (Rosenblatt, 1957) upgrades this to a *threshold logic unit* (TLU): inputs and output are numbers, each input connection carries a weight, the unit computes a weighted sum `z = xᵀw` and applies a *step function* — typically the Heaviside step (0 below zero, 1 at or above) or the sign function. A perceptron is a single layer of TLUs, each connected to every input.

**Intuition.** A neuron is a little voting machine: weigh the evidence, and fire if the total clears a threshold. Even the binary version can compute logic — one wiring gives identity, another AND, another OR, and with an inhibiting connection you get NOT — and these compose into complex logical expressions.

**Notes.** An extra bias feature (`x_0 = 1`) is normally added, supplied by a *bias neuron* that always outputs 1 — it shifts the activation threshold, making the neuron more or less "trigger happy" (quicker or slower to activate) independently of the inputs. A perceptron with several output TLUs is a multi-output classifier (several binary classes at once). *(Hands-On ML)* → Dense layer, Perceptron learning rule, Multi-layer perceptron.

### Dense (fully connected) layer

**Definition.** A layer in which every neuron is connected to every neuron of the previous layer. A whole layer's output for a batch of instances is `φ(XW + b)`: `X` is the input matrix (one row per instance, one column per feature), `W` the weight matrix (one row per input neuron, one column per neuron in the layer), `b` the bias vector (one term per neuron), and `φ` the activation function.

**Intuition.** One matrix multiply computes an entire layer for every instance at once — this is why networks are fast on modern hardware. *Input neurons* are passthrough units that just emit whatever they're fed; together they form the input layer.

**Notes.** For a TLU layer `φ` is a step function; in practice it's ReLU or similar. *(Hands-On ML)* → Artificial neuron and the perceptron, Activation function, Hidden layers and units.

### Perceptron learning rule

**Definition.** A weight update driven by the network's error: `w_{i,j} ← w_{i,j} + η (y_j − ŷ_j) x_i`, where `w_{i,j}` connects input neuron `i` to output neuron `j`, `x_i` is the input value, `ŷ_j` the predicted output, `y_j` the target, and `η` the learning rate.

**Intuition.** A variant of Hebbian learning ("neurons that fire together wire together") that accounts for the mistake made: connections that would have reduced the error get reinforced, in proportion to how wrong the output was and how active the input was.

**Notes.** *(Hands-On ML)* → Artificial neuron and the perceptron, Gradient descent.

### Multi-layer perceptron (MLP) and the XOR problem

**Definition.** An MLP stacks perceptrons: one passthrough *input layer*, one or more *hidden layers* of units, and an *output layer*. Layers near the input are the *lower* layers, those near the output the *upper* layers; every layer except the output includes a bias neuron and is fully connected to the next.

**Intuition.** A single perceptron can only carve a straight boundary, so it famously cannot solve XOR. Stack them and it can — the hidden layer builds intermediate features that make the problem linearly separable at the next level. This stacking is exactly why depth buys expressiveness.

**Notes.** Signal flows one way, input to output, so this is a *feedforward neural network (FNN)*. An FNN with a deep stack of hidden layers is a *deep neural network (DNN)*. *(Hands-On ML)* → Neural networks (feed-forward), Hidden layers and units, Backpropagation.

### Backpropagation

**Definition.** The training algorithm for neural nets, run on one mini-batch at a time for multiple *epochs* (full passes over the training set). Each step: (1) *forward pass* — push the batch through the layers to the output, keeping all intermediate results; (2) measure the output error with a loss function; (3) *backward pass* — apply the chain rule to work out how much each connection contributed to the error, layer by layer back to the input; (4) take a gradient-descent step on all weights using those gradients.

**Intuition.** A forward pass is just prediction with the scratch work saved. The backward pass then assigns blame: the chain rule propagates the error gradient backward through the network — hence the name — so every weight learns how much it was responsible and which way to move. A gradient is a *slope*: given a nudge to this weight, how much does the loss move?

**Notes.** The automatic gradient computation is *automatic differentiation (autodiff)*, or *autograd*; backpropagation uses *reverse-mode autodiff*, which is fast and precise and well suited to functions with many inputs (weights) and few outputs (one loss). Each operation only needs its own *local* derivative, chained with the incoming gradient. A few patterns recur when doing it by hand: a **mean/sum** distributes `1/n` (or 1) to each input; a **broadcast** in the forward pass becomes a **sum** over that axis in the backward; and a value **reused** in several places accumulates the sum of its gradients. A famous simplification: the gradient of softmax + cross-entropy collapses to `softmax(logits)`, minus 1 at the correct class, divided by the batch size — predicted probabilities pulled toward the truth. *(Hands-On ML; Karpathy)* → Gradient descent, Computational graph and autograd, Zeroing gradients, Numerical stability, Weight initialization, Stochastic gradient descent.

### Computational graph and autograd

**Definition.** An *autograd* engine records every operation as a graph: each value stores its data, its running gradient, which values produced it, and the local derivative rule for the operation that made it. Calling `backward()` on the final output sorts the graph topologically, seeds that output's gradient to 1, and walks the nodes in reverse applying each local rule.

**Intuition.** You never derive a formula for the whole network's derivative. You only ever specify the derivative of each small operation; the graph chains them together for you. That's what makes arbitrarily complicated models differentiable "for free."

**Notes.** How coarsely you carve the graph is a free choice — `tanh` can be one fused node or several smaller ones (`exp`, subtract, divide), and the leaf gradients come out identical either way. You only need the local derivative of whatever pieces you define. *(Karpathy)* → Backpropagation, Activation function.

### Zeroing gradients

**Definition.** Gradients *accumulate* — each backward pass adds into the existing `.grad` rather than replacing it — so every parameter's gradient must be reset to zero before each backward pass.

**Intuition.** Accumulation is necessary and correct: if a value feeds into the graph along more than one path, each path contributes and the contributions must sum. (Use a value twice in an addition and its gradient is 1 + 1 = 2, not 1.) The side effect is that stale gradients from the previous iteration linger, so a training loop that forgets to clear them pushes each update with the sum of every gradient so far.

**Notes.** The classic silent bug in a hand-written training loop — the loss still decreases sometimes, just wrongly. *(Karpathy)* → Backpropagation, Gradient descent.

### Numerical stability

**Definition.** Rearranging a computation so intermediate values don't overflow or underflow floating point. The canonical example is the *log-sum-exp* trick in softmax/cross-entropy: subtract the maximum logit before exponentiating, so the largest exponent is 0.

**Intuition.** `exp` of a large logit overflows to infinity and `log(0)` underflows to negative infinity — both wreck the loss. Subtracting the max makes the biggest term `exp(0)=1` and the rest smaller, which changes nothing mathematically (softmax is shift-invariant) but keeps every number in a safe range. This is one reason to call a library `cross_entropy` (which fuses and stabilizes the steps) rather than composing `exp`, `sum`, `log` by hand.

**Notes.** *(Karpathy)* → Softmax, Likelihood and negative log-likelihood, Output layer and loss.

### Weight initialization

**Definition.** Hidden-layer connection weights must be initialized *randomly*, not to a constant such as zero — and at a *scale* that keeps activations well-behaved: divide by `sqrt(fan_in)` (the number of inputs to the unit), times an activation-specific *gain* (`5/3` for tanh). This is *Kaiming* (a.k.a. He) initialization; the Xavier/Glorot variant is similar.

**Intuition.** Two separate points. *Break symmetry:* initialize everything to zero and every neuron in a layer is identical, so backprop updates them identically and they stay identical forever — a layer of hundreds behaves like one neuron. Random values fix this. *Control the scale:* multiplying by a random weight matrix widens the spread of activations, so without the `1/sqrt(fan_in)` factor activations grow (or shrink) layer to layer, pushing a `tanh` into its flat saturated tails where gradients die. Dividing by `sqrt(fan_in)` keeps the spread roughly constant with depth; the gain undoes the squash of the non-linearity.

**Notes.** Also matters at the output layer: scaling the last layer's weights down (so initial logits are near 0) starts training from a near-uniform, low-loss softmax instead of wasting steps un-doing an over-confident random guess. Batch normalization largely removes the sensitivity to all of this. *(Hands-On ML; Karpathy)* → Backpropagation, Activation saturation, Batch normalization, Activation function.

### Activation saturation

**Definition.** When a squashing activation (sigmoid, tanh) is driven far from zero, its output flattens against its asymptote (±1 for tanh) and its derivative goes to ~0 — so gradients through that unit vanish and it stops learning. The ReLU analogue is a *dead neuron* (stuck outputting 0 with zero gradient for all inputs).

**Intuition.** A saturated neuron is pinned to the flat part of its curve: nudging its input barely changes its output, so backprop passes almost no gradient back and the neuron is effectively frozen. A layer full of them learns nothing. Diagnose by histogramming activations — too much mass at ±1 (a "white" saturation map) is the warning sign.

**Notes.** Caused by weights initialized too large or unnormalized pre-activations; fixed by proper Weight initialization or Batch normalization. Leaky ReLU, ELU, and maxout are far less prone to it. Connects to the vanishing-gradient problem in deep nets. *(Karpathy)* → Weight initialization, Batch normalization, Activation function, Vanishing and exploding gradients.

### Batch normalization

**Definition.** A layer, placed right after a linear (or convolutional) layer, that normalizes the pre-activations to zero mean and unit variance *across the current batch*, then re-scales and re-shifts them with two learned parameters, `gamma` (scale) and `beta` (shift). At test time it uses fixed statistics — a running (momentum) average of the batch mean/variance collected during training, or a one-time pass over the training set.

**Intuition.** Instead of hand-tuning initialization so activations stay healthy, just force them healthy at every step: standardize, then let `gamma`/`beta` learn whatever scale and offset are actually useful. This makes deep nets far more forgiving of weight initialization and learning rate. Two knock-on effects: the preceding layer's bias becomes redundant (batchnorm subtracts the mean, cancelling any added constant — `beta` is the effective bias), and because each example is normalized using *its batch's* statistics, its activations jitter slightly depending on which examples share the batch. That coupling is a small random augmentation of the activations, so batchnorm also acts as a mild **regularizer** — at the cost of making predictions subtly batch-dependent (hence the need for running statistics at inference).

**Notes.** Uses the unbiased variance (Bessel's correction, `n−1`) and an `eps` for stability. One of the most impactful training tricks in deep learning; alternatives like layer norm avoid the batch-dependence. *(Hands-On ML; Karpathy)* → Weight initialization, Activation saturation, Neural network regularization, Vanishing and exploding gradients, Layer normalization.

### WaveNet / hierarchical context

**Definition.** An architecture that fuses a sequence *progressively* rather than all at once: instead of flattening the whole context into one vector for a single hidden layer, it combines a few elements at a time (e.g. pairs) and repeats, building a tree — characters → bigrams → 4-grams → 8-grams.

**Intuition.** Squashing an entire long context into one layer forces every position to interact immediately, before any local structure is learned. A hierarchy lets early layers learn short-range patterns and later layers combine those into longer-range ones — a smoother, deeper way to grow the context window. (WaveNet uses this, with dilated convolutions, for raw audio.)

**Notes.** In the makemore build this is `FlattenConsecutive(2)` blocks halving the time dimension each step; a longer `block_size` becomes usable. Cousin of the CNN idea of stacking local receptive fields. *(Karpathy)* → Convolutional neural network, Recurrent neural network, Language model.

### Train / dev / test split

**Definition.** Partition the data three ways: a **training** set to fit parameters, a **dev (validation)** set to choose hyperparameters, and a **test** set touched only once at the very end for an honest final estimate.

**Intuition.** The validation set is how you pick model size, learning rate, embedding dimension, and so on — but tuning against it means you slowly overfit *those choices* to it. A held-back test set you never optimize against catches that. Train ≈ dev loss suggests underfitting (grow the model); a big train-dev gap suggests overfitting.

**Notes.** The deep-learning-practice version of ISL's train/validation idea and cross-validation. Common split: 80/10/10. → Validation set approach (stat), k-fold cross-validation (stat), Overfitting (stat).

### Neural networks (feed-forward)

**Definition.** A model that predicts by feeding inputs through one or more layers of *derived features*. A feed-forward network for `p` inputs has the form `f(X) = β_0 + Σ_k β_k h_k(X)`, where each hidden unit `h_k(X) = g(w_{k0} + Σ_j w_{kj} X_j)` is a non-linear transform `g` of a weighted sum of the inputs.

**Intuition.** The network invents its own intermediate features (the hidden units) — each a squashed weighted mix of the inputs — then combines those features to produce the output. Stacking layers lets later features build on earlier ones; that depth is what "deep" learning means. The "neuron" name is a loose brain analogy.

**Notes.** The raw features form the *input layer*; the derived `h_k` form a *hidden layer*. What sets neural nets apart from other flexible methods is this layered structure. → Hidden layers and units, Activation function.

### Hidden layers and units

**Definition.** The layers of derived features between input and output. Each hidden *unit* is one derived feature; a network can have many units per layer and many layers stacked.

**Intuition.** More units and layers = more capacity to represent complex functions. Modern practice: use *many* units and control overfitting with regularization rather than by keeping the network small.

**Notes.** Number of layers and units per layer are key architecture choices, tuned alongside the regularization knobs. → Neural networks, Dropout.

### Activation function

**Definition.** The non-linear function `g` applied inside each hidden unit. Common choices: the *sigmoid/logistic* `g(z) = 1/(1 + e^{−z})` (output in `(0,1)`), the *hyperbolic tangent* `tanh(z) = 2σ(2z) − 1` (same S-shape, output in `(−1,1)`), and the *ReLU* (rectified linear unit) `g(z) = max(0, z)`. Variants of ReLU include leaky ReLU, PReLU, RReLU, ELU, and SELU.

**Intuition.** Without a non-linearity, stacking linear layers would just give another linear model — the activation is what lets the network bend. ReLU (pass positives, zero out negatives) is the modern default: cheap to compute, and having no maximum output value avoids some gradient problems that saturating S-curves cause. tanh's advantage over the logistic is that its outputs are centered near 0 at the start of training, which often speeds convergence.

**Notes.** ReLU is not differentiable at `z = 0` (the slope jumps, which can make gradient descent bounce) and its derivative is 0 for `z < 0` — in practice it still works very well. A rough practical ordering for hidden layers: **SELU > ELU > leaky ReLU (and variants) > ReLU > tanh > logistic**, though it varies; ELU may beat SELU when the architecture can't self-normalize, and leaky ReLU is preferable when runtime latency matters. Try RReLU if overfitting, PReLU with a very large training set. *(Hands-On ML)* → Neural networks, Vanishing and exploding gradients.

### Output layer and loss

**Definition.** How the final layer and training objective are set. For a quantitative response: a linear output trained with squared-error loss `Σ(y_i − f(x_i))²`. For classification: a *softmax* output producing class probabilities, trained by minimizing the negative multinomial log-likelihood, a.k.a. *cross-entropy*.

**Intuition.** A regression net aims to land near the number (squared error); a classification net aims to put high probability on the true class (cross-entropy), exactly like multinomial logistic regression. Softmax turns the final scores into probabilities summing to 1.

**Notes.** Softmax and cross-entropy are the neural-net versions of ideas already in multinomial logistic regression. → Multinomial logistic regression (stat), Maximum likelihood (stat), Softmax, Likelihood and negative log-likelihood.

### Softmax

**Definition.** The function that turns a vector of real-valued scores (*logits*) into a probability distribution: **exponentiate each, then normalize** by their sum. Output values are positive and sum to 1.

**Intuition.** Exponentiating forces every score positive and blows up the differences (the biggest logit dominates); dividing by the total scales them into probabilities. It's the multi-class stand-in for the logistic function.

**Notes.** Reading exp(logit) as a *count* makes the picture concrete: a count model normalizes raw counts into probabilities, and softmax normalizes exp(logits) the same way — so a logit is a learned log-count. The output layer of a classification net. → Logits, Output layer and loss, Multinomial logistic regression (stat).

### Logits

**Definition.** The raw, un-normalized scores a model outputs before softmax — interpretable as *log-counts*.

**Intuition.** The confusing count↔logit↔probability chain, untangled: a count model has counts `N` → normalize → probabilities. A network outputs logits; `exp(logits)` gives numbers playing the exact role of `N` (positive "counts"); normalizing those is softmax → probabilities. So logits are just the network's learned version of the log of a count table — instead of tallying counts, gradient descent discovers them.

**Notes.** "Log-counts" and "logits" are the same thing in this setting. → Softmax, Output layer and loss.

### One-hot encoding

**Definition.** Representing a categorical value (say index `k` out of `K`) as a length-`K` vector that is 1 in position `k` and 0 everywhere else.

**Intuition.** A raw index is a misleading number — category 5 isn't "half of" category 10, and categories have no order or spacing. One-hot strips out that false arithmetic, giving each category its own independent input slot. Multiplying a one-hot vector by a weight matrix simply selects one row of that matrix.

**Notes.** Standard input encoding for categorical features and for characters/words in language models; dense *embeddings* are the learned, lower-dimensional successor. → Word embeddings, Dummy variables.

### Likelihood and negative log-likelihood

**Definition.** The *likelihood* of the data under a model is the product of the probabilities it assigns to every observed event. The *log-likelihood* is the sum of the logs of those probabilities (`log(a·b·c) = log a + log b + log c`). The *negative log-likelihood (NLL)* is its negation; the *average NLL* divides by the number of events.

**Intuition.** Maximizing likelihood = making the observed data look as probable as possible under the model. Three equivalent moves make it a usable loss: take logs (a stable sum instead of a vanishing product of tiny numbers — and monotonic, so it preserves the ranking), then negate (we want a loss that's *lower when better*, but likelihood runs the other way), then average (a clean, dataset-size-independent number). The best possible average NLL is 0 — perfect confidence, always correct.

**Notes.** Average NLL over a softmax output is exactly *cross-entropy loss*. Minimizing NLL is the loss-function face of maximum likelihood. → Maximum likelihood (stat), Softmax, Output layer and loss.

### Language model

**Definition.** A model of the probability of the next token in a sequence given the previous ones. A *character-level* language model does this over characters, treating each character as a training example and learning to predict the next character. An *n-gram* model conditions on the previous `n − 1` tokens; a *bigram* model (`n = 2`) uses only the single previous token.

**Intuition.** Learn "what usually comes next," then generate by sampling one token at a time and feeding it back in. A bigram model has almost no context — it's the simplest thing that works — while longer context (more history, or an RNN's hidden state) makes for a stronger model.

**Notes.** The count-based bigram model and its neural-network twin compute the same thing: the network's learned weights are the equivalent of the (log) count table. Beyond bigrams, a fixed **context window** of the previous few characters can be embedded and fed through an MLP (the Bengio 2003 design), or fused hierarchically (WaveNet); an RNN instead carries unbounded context in a hidden state. → Logits, One-hot encoding, Embedding layer, WaveNet / hierarchical context, Recurrent neural network.

### Broadcasting

**Definition.** The rule that lets element-wise operations combine tensors of different shapes by *virtually* stretching the smaller one — e.g. dividing a `(27, 27)` matrix by a `(27, 1)` column of row-sums copies that column across all columns, then divides element-wise. No data is actually duplicated.

**Intuition.** A convenience that saves loops, but a notorious source of silent bugs: when normalizing a matrix's rows, the sum must keep shape `(27, 1)` (`keepdims=True`) so it broadcasts *down each row*; a bare `(27,)` broadcasts across the wrong axis and normalizes columns instead — no error, just wrong numbers.

**Notes.** *(Karpathy)* → Feature scaling.

### Convolutional neural network (CNN)

**Definition.** A network specialized for images, combining *convolution* layers (which slide filters across the image) and *pooling* layers (which downsample).

**Intuition.** Rather than wire every pixel to every unit, a CNN slides small filters over the image to detect local patterns (edges, textures), then builds from simple to complex features layer by layer. It exploits the spatial structure of images the way RNNs exploit sequence order.

**Notes.** Color images have three channels (R, G, B); a filter carries one sub-filter per channel and sums their results into one feature map. Using `K` filters yields `K` feature maps, stacked as the next layer's input. → Convolution filter, Pooling.

### Convolution filter

**Definition.** A small weight matrix slid over every patch of an image; at each position it multiplies element-wise with the patch and sums to one number. Sweeping it over all patches produces a *convolved feature map*.

**Intuition.** The filter is a little pattern detector — its output lights up where the image locally resembles the filter. Different filters catch different features. A ReLU is typically applied after (sometimes called the *detector* step).

**Notes.** Filter weights are *learned*, not hand-set. → Convolutional neural network, Activation function.

### Pooling

**Definition.** A downsampling step; *max pooling* replaces each non-overlapping block (e.g. 2×2) with its maximum value.

**Intuition.** Shrinks the feature map (halving each dimension for 2×2) and grants some *location invariance* — the network registers that a feature is present in a region, not its exact pixel. Fewer parameters, more robustness.

**Notes.** → Convolutional neural network.

### Data augmentation

**Definition.** Expanding the training set by adding randomly distorted copies of each example (for images: small rotations, shifts, flips) that don't change the label.

**Intuition.** Teach the network that a cat is still a cat when nudged or mirrored — free extra training data that improves generalization and fights overfitting.

**Notes.** A regularizer tailored to structured inputs like images. → Dropout.

### Bag-of-words

**Definition.** A simple way to turn a document into features: for a dictionary of `M` words, represent each document as a length-`M` binary vector marking which words are present (1) or absent (0).

**Intuition.** Ignore grammar and order; just record which words show up. Crude but often effective for text classification. The vector is huge, so the dictionary is capped (e.g. the 10,000 most frequent words).

**Notes.** Throws away word order — RNNs and embeddings capture what bag-of-words discards. → Recurrent neural network, Word embeddings.

### Recurrent neural network (RNN)

**Definition.** A network for *sequential* inputs `X = {X_1, …, X_L}` (words, time steps). It processes the sequence one element at a time, maintaining a hidden activation `A_ℓ` updated from the current input `X_ℓ` and the previous activation `A_{ℓ−1}`.

**Intuition.** The hidden state is a running memory: each step folds in the new element while carrying context forward from earlier ones. This lets *order* matter — what bag-of-words ignores — the way CNNs let spatial layout matter.

**Notes.** Words are fed as one-hot vectors or, better, embeddings. → Word embeddings, LSTM.

### Embedding layer

**Definition.** A learned lookup table that maps each item of a discrete vocabulary (word, character, category) to a dense, low-dimensional vector. Stored as a matrix `C` with one row per vocabulary item; "embedding" a batch of indices is just row-indexing, `C[X]`. The vectors are parameters, trained by backprop like any other weights. Pretrained word embeddings include *word2vec* and *GloVe*.

**Intuition.** A one-hot vector treats every item as equally unrelated to every other; an embedding places them in a space where similar items sit close together, and the model learns that geometry from the task. Indexing `C[i]` is the efficient equivalent of `one_hot(i) @ C` — it selects one row without building the one-hot. Far richer input than raw indices, and (for pretrained vectors) reusable across tasks.

**Notes.** In a character-level model the embedding table is small (e.g. 27 characters × 10 dims) and learned from scratch; you can plot a 2-D one and watch related characters cluster. The successor to one-hot input; the input stage of most language models. → One-hot encoding, Recurrent neural network, Language model.

### LSTM

**Definition.** *Long short-term memory* — an elaboration of the RNN that maintains two tracks of hidden activations, so a unit can draw on context from both the recent and the more distant past.

**Intuition.** Plain RNNs struggle to remember information from far back in a sequence. LSTM adds a longer-memory channel alongside the short-term one, so important early signal isn't washed out.

**Notes.** → Recurrent neural network.

### Autoregressive models and autocorrelation

**Definition.** In time-series data, observations aren't independent — they show *autocorrelation* (correlation between `v_t` and its lagged value `v_{t−ℓ}`). An order-`L` *autoregressive* model, AR(L), regresses each value on its previous `L` values: `v_t = β_0 + β_1 v_{t−1} + … + β_L v_{t−L}`.

**Intuition.** Today resembles recent days, so predict from the recent past. Chop the series into overlapping windows of length `L` (the *lag*) and fit. An RNN and an AR model use the same windows; the AR model *flattens* each window into one predictor vector, while the RNN processes it in order with shared weights and a hidden state, adding non-linearity.

**Notes.** Autocorrelation is also the "correlated errors" caveat for ordinary regression. → Potential problems in linear regression (stat), Recurrent neural network.

### Gradient descent

**Definition.** The iterative method for fitting a network: start from a guess for all parameters `θ`, then repeatedly nudge `θ` a small step in the direction that most decreases the loss `R(θ)`, until it stops improving. The step size is the *learning rate* `ρ`.

**Intuition.** Roll downhill on the loss surface. The gradient points uphill, so step against it; take small enough steps and the loss keeps dropping. Reach a spot where the gradient is zero and you've found a minimum. Each component of the gradient is a *partial derivative* — "if I nudge parameter `θ_j` alone, how much does the loss change?" Picture standing on a hillside and asking the slope facing east, then facing north, once per dimension.

**Notes.** Network loss surfaces are non-convex, so gradient descent can settle in a *local* rather than *global* minimum. "Slow learning" — small steps plus early stopping when overfitting appears — is itself a safeguard. A generic optimizer, not neural-net-specific: it solves a wide range of problems. → Stochastic gradient descent, Double descent, Early stopping.

### Stochastic gradient descent (SGD)

**Definition.** A faster variant of gradient descent that computes each step's gradient on a small random *minibatch* of the data rather than all of it. Three points on the same spectrum: *batch* GD uses the whole training set per step; *stochastic* GD uses a single randomly chosen instance; *mini-batch* GD uses a small random subset (e.g. 32).

**Intuition.** Estimating the downhill direction from a small sample is noisier but far cheaper per step, so you take many more steps in the same time — usually reaching a good solution faster. Batch GD's problem is exactly this: using every instance for every step makes it very slow on large training sets. Pure stochastic GD sits at the other extreme — barely any data per iteration, so it's fast and can train on huge sets since only one instance need be held at a time; the cost is a noisy, jittery descent path. *(Hands-On ML)*

**Notes.** The standard optimizer for large networks. Key knobs: batch size and number of *epochs* (full passes over the training set). → Gradient descent, Backpropagation, Learning-rate schedule.

### Learning-rate schedule

**Definition.** Varying the learning rate (step size) over the course of training rather than holding it fixed — most commonly *decay*, where the step size shrinks at higher epochs.

**Intuition.** Big steps early cover ground fast toward the minimum; small steps late stop you from bouncing around it and let you settle in precisely. A constant rate forces a bad compromise between the two.

**Notes.** One of the more impactful training knobs in deep learning. To find a sane range in the first place, sweep the rate exponentially (e.g. 1e-3 up to 1) over steps and watch where the loss falls fastest — too low barely moves, too high diverges. *(Karpathy)* → Gradient descent, Stochastic gradient descent.

### Adam and AdamW

**Definition.** *Adam* (adaptive moment estimation) is a gradient-descent optimizer that keeps a per-parameter running average of the gradient (first moment, a momentum term) and of its square (second moment), and scales each parameter's step by them — so every parameter gets its own adaptive learning rate. *AdamW* is the variant that applies weight decay (L2 shrinkage) *decoupled* from the gradient update rather than folded into it.

**Intuition.** Plain SGD uses one global step size for every parameter; Adam gives parameters with consistently large gradients smaller steps and rarely-updated ones larger steps, which makes it fast and robust with little tuning — the default optimizer for transformers. AdamW's decoupled weight decay fixes a subtlety where Adam's per-parameter scaling distorts ordinary L2 regularization, so the shrinkage acts as intended.

**Notes.** A common default learning rate is ~3e-4. The GPT build trains with `torch.optim.AdamW(model.parameters(), lr=...)`. *(Karpathy)* → Stochastic gradient descent (SGD), Learning-rate schedule, Neural network regularization.

### Dropout

**Definition.** A regularization method that randomly removes a fraction `φ` of the units in a layer during each training update (setting their activations to zero), scaling the survivors up by `1/(1−φ)` to compensate.

**Intuition.** By randomly knocking out units, no single unit can become indispensable, so the network spreads its representation and generalizes better. Inspired by random forests' trick of randomly restricting features; applied fresh for each training example.

**Notes.** Similar in spirit to ridge regularization; ridge/lasso penalties are also applied to network weights. One of the most popular regularizers for deep nets — even strong networks have gained 1–2 percentage points of accuracy from it, which at 95% accuracy means cutting the error rate by nearly 40%. *(Hands-On ML)* → Random forests, Ridge regression (stat), MC dropout, Neural network regularization.

### Neural network regularization

**Definition.** The family of techniques that constrain a network's freedom to fit noise: *early stopping*, *batch normalization* (as a side effect), *L1 and L2 penalties* on the connection weights (typically not the biases), *dropout*, and *max-norm regularization*.

**Intuition.** Deep networks routinely carry tens of thousands to millions of parameters — enough freedom to fit almost any dataset, and therefore enough to memorize noise. (Von Neumann's line: with four parameters he could fit an elephant, with five make its trunk wiggle; with thousands you can fit the whole zoo.) Regularization is what buys back generalization.

**Notes.** L1/L2 on network weights is the same idea as the lasso and ridge penalties for linear models: the penalty is computed each training step and added to the loss. In a softmax classifier, an L2 penalty (`+λ·mean(W²)`) pulls the weights toward 0, which pulls the logits toward 0 and the class probabilities toward *uniform* — so weight decay acts as a smoothing pressure, the gradient-based twin of adding fake counts to a count model. `λ` sets how much: too large and every prediction blurs toward uniform. *(Hands-On ML; Karpathy)* → Dropout, Max-norm regularization, Early stopping, Ridge regression (stat), The lasso (stat), Softmax.

### Max-norm regularization

**Definition.** For each neuron, constrain the incoming connection weights so that `‖w‖₂ ≤ r`, where `r` is the max-norm hyperparameter.

**Intuition.** Rather than penalizing large weights in the loss, simply cap them: after each update, if a neuron's weight vector has grown past the ceiling, rescale it back down. A hard constraint instead of a soft penalty.

**Notes.** *(Hands-On ML)* → Neural network regularization.

### MC dropout

**Definition.** *Monte-Carlo dropout*: leave dropout switched on at prediction time, run the same input through the network many times, and average the resulting predictions.

**Intuition.** Each pass with dropout active is a slightly different network, so the spread of their predictions is a usable measure of the model's *uncertainty* — and averaging them typically improves accuracy too. It works on an already-trained dropout model with no retraining and no change to the architecture.

**Notes.** A 2016 result (Gal and Ghahramani) also established a connection between dropout networks and approximate Bayesian inference, giving dropout a firmer theoretical grounding. *(Hands-On ML)* → Dropout.

### Early stopping

**Definition.** Halt training once validation performance stops improving, rather than running to convergence on the training loss.

**Intuition.** Training error keeps falling long after test error has bottomed out; early stopping catches the model at the bottom of that U instead of letting it slide into overfitting. Cheap, effective, and it needs no change to the model.

**Notes.** One of the best regularizers for neural nets, and the practical face of the "slow learning" safeguard. *(Hands-On ML)* → Neural network regularization, Gradient descent, Training MSE vs test MSE (stat).

### Vanishing and exploding gradients

**Definition.** The problem that gradients propagated backward through a deep network can shrink toward zero (so lower layers barely train) or blow up (so training diverges).

**Intuition.** Each backward step multiplies by a layer's local derivative; do that through many layers and the product can decay or explode geometrically. Saturating activations like the logistic make it worse, since their derivative is near zero over much of their range — a large part of why ReLU and its variants, with no upper saturation, became the default. Batch normalization was designed to address this too.

**Notes.** *(Hands-On ML)* → Activation function, Backpropagation.

### Double descent

**Definition.** A phenomenon where, as model flexibility increases past the point of *interpolation* (zero training error), test error — after the usual U-shaped rise — can fall *again*.

**Intuition.** Classic bias–variance says test error is U-shaped and interpolating the data is bad, and that holds *up to* the interpolation threshold. But push flexibility even further and, in some settings, test error descends a second time — so a model that fits training data perfectly can sometimes beat a slightly less flexible one. Hence "double" descent.

**Notes.** A modern refinement of, not a contradiction to, the bias–variance trade-off — the trade-off governs behavior up to interpolation. Helps explain why huge over-parameterized networks can still generalize. → Bias–variance trade-off (stat), Overfitting (stat).

---

## Transformers and Large Language Models *(Karpathy)*

### Tokenization and byte-pair encoding (BPE)

**Definition.** Turning raw text into a sequence of integer *tokens* (and back) via a fixed vocabulary. A *character-level* tokenizer uses one token per character (tiny vocab, long sequences). Real LLMs use *subword* tokenizers, chiefly *byte-pair encoding (BPE)*: start from the 256 raw *UTF-8 bytes* as the base vocabulary, then repeatedly find the most frequent adjacent token pair and merge it into a new token, for a fixed number of merges.

**Intuition.** A lossless, reversible mapping — `encode` text → ints, `decode` ints → text, with `decode(encode(s)) == s`. The core trade-off is vocabulary size vs sequence length: bytes/characters give a small vocab but long sequences (so a fixed context window covers less text); merging frequent pairs shortens sequences at the cost of a bigger vocab. *Training* the tokenizer = running the merge loop to learn a `merges` table; *encoding* re-applies those merges in the order learned (earliest merge id first) until none apply; *decoding* concatenates each token's stored bytes and UTF-8-decodes.

**Notes.** The tokenizer is a *separate module* from the model — its own training data, its own BPE pass — and the model only ever sees token ids. Working on raw UTF-8 bytes means any string is representable (no out-of-vocabulary). GPT-3 vocab 50,257; LLaMA 32,000. → Regex pre-tokenization, Special tokens, tiktoken vs SentencePiece, Tokenization failure modes, One-hot encoding, Embedding layer, Language model.

### Regex pre-tokenization

**Definition.** Splitting text with a regular expression *before* BPE, so merges are only ever computed *within* each chunk, never across chunk boundaries. GPT-2's pattern breaks text into words, numbers, punctuation runs, and whitespace (using Unicode classes `\p{L}` letters, `\p{N}` numbers), and keeps a leading space attached to the following word (` ?\p{L}+`).

**Intuition.** Left unconstrained, BPE would merge across category and word boundaries — fusing `dog` with a trailing `.` or a space — wasting vocabulary and hurting generalization. Pre-splitting forces merges to respect linguistic boundaries. It's also why tokens so often begin with a space.

**Notes.** GPT-4's tokenizer revised this pattern (and, unlike GPT-2, does merge runs of spaces). Requires the `regex` module for `\p{...}` classes. → Tokenization and byte-pair encoding (BPE).

### Special tokens

**Definition.** Reserved tokens added to the vocabulary *outside* the BPE merges, with fixed ids, to mark structure rather than content — e.g. GPT-2's single `<|endoftext|>` marking document boundaries; chat models add turn-boundary tokens.

**Intuition.** The model needs signals like "document ends here" or "user turn ends, assistant begins" that can't be confused with ordinary text. Because they're handled specially, a literal `<|endoftext|>` typed into the input can be intercepted as a control token instead of those characters.

**Notes.** GPT-2's vocab is 256 byte tokens + 50,000 merges + 1 special = 50,257. Adding new special tokens means growing the embedding and output layers. → Adding vocabulary after training, Tokenization and byte-pair encoding (BPE), Language model.

### tiktoken vs SentencePiece

**Definition.** Two production tokenizers. *tiktoken* (OpenAI) encodes text to UTF-8 bytes and runs BPE on the *bytes* (inference-only). *SentencePiece* (Google; used by Llama/Mistral) runs BPE on the Unicode *code points* directly, and can both train and infer.

**Intuition.** The difference is *what* BPE operates on: bytes (tiktoken) vs code points (SentencePiece). SentencePiece handles rare code points via `character_coverage` plus `byte_fallback` — very rare characters either map to an `UNK` token or fall back to raw UTF-8 byte tokens. Its `add_dummy_prefix` prepends a space so a word tokenizes the same at the start of a sentence as mid-sentence.

**Notes.** tiktoken can't train a new vocabulary; SentencePiece is the go-to when you need to *train* a tokenizer. → Tokenization and byte-pair encoding (BPE).

### Tokenization failure modes

**Definition.** Many LLM quirks trace to tokenization rather than the model: weak spelling and character-level tasks (a token isn't its letters), shaky digit arithmetic (numbers split inconsistently), worse non-English performance (more tokens per word → less effective context), trailing-whitespace sensitivity, and "glitch tokens" like `SolidGoldMagikarp` (rare tokens the model barely trained on).

**Intuition.** The model reasons over tokens, not characters, so anything needing sub-token structure — spelling, reversing a string, per-digit arithmetic — is handicapped by how the text was chunked. Tokens that appeared in the *tokenizer's* training but hardly in the *model's* trigger erratic behavior when invoked.

**Notes.** Practical upshots: prefer formats that tokenize compactly (sometimes YAML over JSON), and watch for whitespace and special-token pitfalls. → Tokenization and byte-pair encoding (BPE).

### Adding vocabulary after training

**Definition.** Introducing new tokens (new special tokens, domain terms, a new language) into an already-trained model by *extending* the token-embedding matrix and the output (unembedding) layer with fresh rows, then training *only* those new rows while freezing all existing weights.

**Intuition.** The old tokens already have good embeddings you don't want to disturb; the new rows start random and must catch up. Freezing everything else and training just the added embeddings teaches the new tokens cheaply, without degrading what the model already knows — a minimal, targeted fine-tune.

**Notes.** Common when adding control tokens for fine-tuning or adapting to a new domain. *(Karpathy)* → Special tokens, Embedding layer, Parameter-efficient fine-tuning (LoRA / PEFT).

### Self-attention

**Definition.** A layer that lets each position in a sequence gather information from other positions with data-dependent weights. Every token emits three vectors from learned linear maps of its embedding: a *query* (what am I looking for), a *key* (what do I contain), and a *value* (what I'll pass on if attended to). The weight from token `i` to token `j` is the (scaled, softmaxed) dot product of `i`'s query with `j`'s key; the output at `i` is the weighted sum of all values: `out = softmax(Q Kᵀ / sqrt(d)) V`.

**Intuition.** A *communication mechanism*: picture tokens as nodes in a directed graph that aggregate information from the nodes pointing to them, via a weighted sum whose weights depend on the data. A token's query "asks" for certain content; tokens whose keys match get high weight, and their values flow into the output. Because it's just dot products over a *set* of vectors, attention has no built-in sense of position — order must be injected separately (→ Positional encoding). Keeping *value* separate from the raw embedding lets a token expose something different from what it *is*.

**Notes.** "Self" means Q, K, V all come from the same sequence `x`; when K and V come from a different source it's *cross-attention*. Each batch example is processed completely independently. → Scaled dot-product attention, Causal (masked) self-attention, Multi-head attention, Encoder, decoder, and cross-attention.

### Scaled dot-product attention

**Definition.** The attention scores are divided by `sqrt(head_size)` before the softmax: `wei = (Q Kᵀ) · head_size^(−1/2)`.

**Intuition.** If Q and K have unit variance, their dot product over `head_size` dimensions has variance ≈ `head_size`, so raw scores grow large as the head widens. Feeding large-magnitude scores into softmax makes it *peaky* — it converges toward a one-hot vector that attends to a single token, starving the other positions of gradient (the same saturation problem squashing activations have). Dividing by `sqrt(head_size)` keeps scores unit-variance, so softmax stays *diffuse* early in training and every token can contribute.

**Notes.** Matters most at initialization, when a saturated softmax would pass almost no gradient. → Activation saturation, Softmax, Self-attention.

### Causal (masked) self-attention

**Definition.** In an autoregressive language model, a token may attend only to itself and earlier positions, never future ones. Implemented by setting the upper-triangular entries of the score matrix to `−inf` before the softmax (a lower-triangular mask), so their weights become 0.

**Intuition.** When predicting the next token, letting a position see the future would leak the answer, so the mask forces position `t`'s output to depend only on positions `≤ t`. Mechanically: build a lower-triangular ones matrix, `masked_fill` the zeros with `−inf`, then softmax — the `−inf` entries exponentiate to 0, giving a valid weighted average over the past only. An early toy form of the same idea is a lower-triangular matrix of *uniform* averaging weights `@ x`; attention just replaces the uniform weights with learned, data-dependent ones.

**Notes.** This triangular masking is what makes a block a *decoder* block; deleting the mask lets all tokens see each other (an *encoder* block). → Self-attention, Encoder, decoder, and cross-attention.

### Multi-head attention

**Definition.** Run several self-attention *heads* in parallel, each with its own smaller Q/K/V projections (`head_size = n_embd / n_heads`), then concatenate their outputs and pass them through a linear *projection* back to the embedding dimension.

**Intuition.** One head can track one kind of relationship; multiple heads let the model attend to several different things at once, each in its own subspace (e.g. one head tracking the previous character, another matching vowels). Concatenating and projecting recombines these independent "communication channels" into one representation. Splitting the embedding across heads keeps total compute the same as one full-width head.

**Notes.** The output projection is what lets the concatenated per-head results be added back into the residual stream. → Self-attention, Transformer block, Residual (skip) connections.

### Positional encoding

**Definition.** Extra information added to token embeddings so the model knows each token's *position*, since attention itself is order-agnostic. In the GPT build this is a learned *positional embedding table* indexed by position `0…T−1`, added to the token embedding: `x = tok_emb + pos_emb`.

**Intuition.** Attention treats its inputs as an unordered set, so "the cat sat" and "sat the cat" would look identical without positional information. Giving each slot its own learned vector lets the model recover word order. (The original transformer used fixed sinusoidal encodings; GPT learns them instead.)

**Notes.** Caps the usable context length at the table size (`block_size`). → Self-attention, Embedding layer.

### Residual (skip) connections

**Definition.** Adding a sub-layer's input back to its output — `x = x + sublayer(x)` — so each block computes a *change* to a running representation rather than replacing it. Used around both the attention and feed-forward sub-blocks.

**Intuition.** The additions create a "residual highway" straight from input to output. During backprop, gradient flows unimpeded down this highway (addition distributes gradient equally to both branches), so even very deep stacks train — sidestepping vanishing gradients. Each block only has to learn a small correction to the residual stream, and at initialization the sub-blocks contribute little, so the network starts near the identity.

**Notes.** Introduced by ResNets; essential for training deep transformers. → Vanishing and exploding gradients, Transformer block, Backpropagation.

### Layer normalization

**Definition.** Normalizes each individual example's activation vector to zero mean and unit variance *across its features* (not across the batch), then applies a learned scale `gamma` and shift `beta`. Same formula as batch norm, but reduced over the feature dimension per token rather than over the batch per feature.

**Intuition.** Batch norm's statistics couple examples in a batch together (an example's output depends on its batch-mates), which is awkward for variable-length sequences and small batches. Layer norm normalizes each token's own feature vector, so there's *no cross-example dependence* and train/test behave identically — no running statistics needed. It keeps activations well-scaled through a deep stack.

**Notes.** In modern (*pre-norm*) transformer blocks, layer norm is applied *before* each sub-block rather than after. Its per-token, batch-independent design is exactly why transformers prefer it over batch norm. → Batch normalization, Transformer block.

### Transformer feed-forward network

**Definition.** A small per-token MLP applied after attention: `Linear(n_embd → 4·n_embd) → ReLU → Linear(4·n_embd → n_embd)`, plus dropout. The 4× hidden expansion follows the original paper.

**Intuition.** Attention is where tokens *communicate* (gather information); the feed-forward network is where each token then *thinks* on what it gathered, independently of the others. Widening to 4× gives the non-linearity room to compute richer features before projecting back down to the residual dimension.

**Notes.** The second linear (`4·n_embd → n_embd`) projects back to the embedding size so the result can be added into the residual stream. Applied identically and independently to every position. → Transformer block, Multi-head attention, Activation function.

### Transformer block

**Definition.** The repeating unit of a transformer: multi-head self-attention followed by a feed-forward network, each wrapped in a residual connection and preceded by layer norm — `x = x + sa(ln1(x)); x = x + ffwd(ln2(x))`. A model stacks many such blocks.

**Intuition.** *Communication followed by computation*: the attention sub-block lets tokens exchange information, then the feed-forward sub-block lets each token process it. Residual connections plus pre-block layer norm are what make a deep stack of these trainable.

**Notes.** GPT interleaves `n_layer` of these, each preserving the `(B, T, n_embd)` shape. The pre-norm placement (LN *before* each sub-block) departs from the original paper's post-norm. → Multi-head attention, Transformer feed-forward network, Residual (skip) connections, Layer normalization.

### Encoder, decoder, and cross-attention

**Definition.** The original transformer has two stacks: an *encoder* (bidirectional self-attention — every token sees every other) and a *decoder* (masked self-attention plus a *cross-attention* sub-layer whose queries come from the decoder but whose keys and values come from the encoder's output). *Cross-attention* is attention where Q is produced from one sequence and K, V from another.

**Intuition.** In a translation model, the encoder reads the whole source sentence into rich vectors; the decoder then generates the target one token at a time, using masked self-attention over what it has produced so far *and* cross-attention to consult the encoded source. Self-attention (K, V, Q from the same place) is a token looking at its own sequence; cross-attention is a token looking at an external, fully-visible source.

**Notes.** *Correction to the highlight:* a **GPT is decoder-only** — an encoder-free stack of masked-self-attention blocks, and therefore has **no cross-attention at all**. The "encoder + decoder + cross-attention" picture is the *original* (Vaswani) encoder-decoder transformer used for tasks like translation, not GPT. GPT's blocks also drop the cross-attention sub-layer that the original decoder block has, leaving just masked self-attention + feed-forward. (Encoder-only models like BERT keep the *unmasked* self-attention and drop the decoder.) → Causal (masked) self-attention, Self-attention, Decoder-only transformer (GPT).

### Decoder-only transformer (GPT)

**Definition.** The full generative model: token embedding + positional embedding → a stack of `n_layer` transformer blocks (masked multi-head attention + feed-forward) → a final layer norm → a linear `lm_head` projecting to vocabulary logits. Trained with cross-entropy on next-token prediction; generates by repeatedly sampling the next token and appending it.

**Intuition.** Every piece assembled: embed tokens and positions, let them communicate causally and compute through many blocks, then read out a distribution over the next token. Generation crops the running context to the last `block_size` tokens, takes the logits at the final position, softmaxes, and samples — autoregressively.

**Notes.** This is the architecture behind GPT-2/3; scaling the *same* design (more layers, wider embeddings, more data) is most of what separates the toy build from GPT-3 (96 layers, ~175B parameters, 50K vocab, 2048 context). → Transformer block, Causal (masked) self-attention, Language model, GPT training pipeline.

### GPT training pipeline

**Definition.** The four-stage recipe that turns raw text into an assistant: *pretraining* (next-token prediction on a huge low-quality corpus → a *base model*), *supervised fine-tuning (SFT)* (train on a small, high-quality set of ideal prompt/response demonstrations → an *SFT model*), *reward modeling (RM)* (train a model to score responses from human preference comparisons → an *RM*), and *reinforcement learning (RLHF)* (optimize the SFT model to produce responses the RM scores highly → the deployed *RL model*).

**Intuition.** Pretraining absorbs essentially all the knowledge (thousands of GPUs, months); the later stages are comparatively cheap alignment steps that shape *how* the model uses that knowledge — turning a document-completer into a helpful assistant. The data flips from "huge quantity, low quality" (pretraining) to "low quantity, high quality" (the fine-tuning stages).

**Notes.** ChatGPT and Claude are RLHF models; base models like GPT/LLaMA/PaLM are pretraining-only. → Pretraining and base models, Supervised fine-tuning (SFT), Reward modeling and RLHF.

### Pretraining and base models

**Definition.** Training a transformer from scratch on trillions of words of internet text by next-token prediction. The result is a *base model* — a powerful next-token predictor, not an assistant.

**Intuition.** A base model only "wants" to complete documents; ask it a question and it may reply with *more questions*, because that's what its training text looks like. It can still be *coerced* into tasks by making the prompt look like a document whose natural continuation is the answer (few-shot examples, or a `Q:`/`A:` format). Pretraining learns broad, general-purpose representations that transfer to many downstream tasks.

**Notes.** Order-of-magnitude cost: thousands of GPUs, months, millions of dollars. Base models are the reusable foundation the alignment stages build on. → GPT training pipeline, In-context learning (zero/one/few-shot), Supervised fine-tuning (SFT).

### Supervised fine-tuning (SFT)

**Definition.** Continue training the base model (same next-token objective) on a small, curated dataset of high-quality prompt → ideal-response pairs written by contractors following detailed labeling instructions.

**Intuition.** Same algorithm as pretraining, different data: by imitating thousands of exemplary assistant responses, the model learns the *format and manner* of a helpful assistant rather than raw document completion. Quality matters far more than quantity here (~10K–100K examples).

**Notes.** SFT alone yields a usable assistant and is achievable without huge resources; RLHF is the research-heavier stage layered on top. → GPT training pipeline, Reward modeling and RLHF.

### Reward modeling and RLHF

**Definition.** *Reward modeling:* collect human *comparisons* ranking several model responses to the same prompt, and train a model to predict a scalar reward (read out at a special token) consistent with those preferences. *RLHF:* use reinforcement learning (PPO) to update the SFT model so its sampled responses score highly under the reward model, weighting each token's update by the reward-derived *advantage*.

**Intuition.** Judging which of two answers is better is far easier and more reliable for humans than writing the ideal answer, so preferences scale better than demonstrations. The reward model turns those preferences into a trainable signal; RL then pushes the policy toward high-reward completions — boosting the probabilities of tokens in good responses and suppressing those in bad ones.

**Notes.** Produces the deployed model (ChatGPT, Claude). RLHF is still research territory relative to SFT. → GPT training pipeline, Mode collapse, Supervised fine-tuning (SFT).

### Mode collapse

**Definition.** The loss of output diversity (entropy) that RLHF tends to cause: the aligned model confidently emits a few near-identical phrasings where the base model had a broad distribution.

**Intuition.** RLHF concentrates probability on the responses the reward model likes, so the model stops exploring alternatives. This is why *base* models can beat RLHF models at tasks needing variety — e.g. "generate 100 different names" — where the goal is diversity, not a single best answer.

**Notes.** A concrete cost of alignment; choose base vs aligned by whether the task wants one good answer or many varied ones. → Reward modeling and RLHF, Pretraining and base models.

### In-context learning (zero/one/few-shot)

**Definition.** Getting a model to perform a task purely from the prompt, with *no gradient updates*: *zero-shot* gives only a task description, *one-shot* adds a single worked example, *few-shot* adds several. "Prompting over fine-tuning."

**Intuition.** A capable base model can infer the pattern from a handful of in-prompt examples and continue it — so you often get task-specific behavior without training anything, just by arranging the context. GPT-2/3 popularized this as an alternative to fine-tuning a separate model per task.

**Notes.** Cheap and immediate, but limited by the context window and generally weaker than true fine-tuning on hard tasks. → Language model, Pretraining and base models, Parameter-efficient fine-tuning (LoRA / PEFT).

### Chain-of-thought prompting

**Definition.** Prompting the model to produce intermediate reasoning steps before its final answer (e.g. appending "Let's think step by step"), rather than answering immediately.

**Intuition.** A transformer spends the *same* small amount of compute per token and has no internal scratchpad, so hard problems need their reasoning *spread out over tokens* — the tokens themselves are the working memory. Writing out the steps gives the model room to compute the answer instead of guessing it in one shot. Few-shot CoT shows worked reasoning examples; zero-shot CoT just adds the trigger phrase.

**Notes.** Large accuracy gains on multi-step (e.g. arithmetic) problems. Related lever: because LLMs imitate a *range* of quality, explicitly asking for competent/expert reasoning ("condition on good performance") tends to help. → Self-consistency and ensembling attempts, In-context learning (zero/one/few-shot).

### Self-consistency and ensembling attempts

**Definition.** Sample several independent chain-of-thought solutions to the same problem and take the *majority* final answer instead of trusting a single pass.

**Intuition.** Sampling can get "unlucky" and commit to a bad reasoning path that the model is then stuck with. Generating multiple diverse paths and marginalizing over them (majority vote on the final answer) recovers accuracy — an ensemble over reasoning traces.

**Notes.** Trades compute for reliability. → Chain-of-thought prompting, Tree of thoughts.

### Tree of thoughts

**Definition.** A deliberate problem-solving strategy that expands a *tree* of intermediate thoughts, evaluates them, and searches — keeping promising branches, pruning bad ones — rather than committing to one linear chain.

**Intuition.** The "System 2" (slow, deliberate) counterpart to a single fast pass: explore several partial solutions, score them, and expand the best — analogous to the lookahead search in AlphaGo. More capable than chain-of-thought or self-consistency on problems that benefit from planning and backtracking.

**Notes.** Part of a broader shift from one-shot Q&A toward search and planning over model outputs. → Chain-of-thought prompting, Self-consistency and ensembling attempts, LLM agents (chains, ReAct).

### LLM agents (chains, ReAct)

**Definition.** Wrapping an LLM in a loop that interleaves reasoning with *actions* (tool calls, searches) and observations, rather than answering in one turn. *ReAct* alternates Thought → Action → Observation steps; *AutoGPT*-style systems add task queues and memory to pursue a goal semi-autonomously.

**Intuition.** Think pipelines, state machines, and agents instead of single Q&A: the model plans, takes an action in the world, reads the result, and continues — letting it use tools and decompose long tasks.

**Notes.** Powerful but brittle; the practical recommendation is copilots (human-in-the-loop) over fully autonomous agents. → Tree of thoughts, Retrieval-augmented generation (RAG).

### Retrieval-augmented generation (RAG)

**Definition.** Fetch task-relevant documents at query time and pack them into the context window so the model can condition on them, rather than relying only on what's baked into its weights.

**Intuition.** The context window is the model's "working memory," and it's near-perfect *within* that window — so loading the right reference text into it sharpens answers and supplies facts the weights lack. Emerging recipe: split documents into chunks, embed the chunks into a vector store, retrieve the chunks most similar to the query, and insert them into the prompt.

**Notes.** Sits on a spectrum between pure retrieval (search engines) and pure parametric memory (the raw LLM). Addresses knowledge cutoffs and hallucination. → LLM agents (chains, ReAct), Embedding layer, LLM limitations and safe use.

### Constrained prompting

**Definition.** "Prompting languages" that interleave free generation with enforced structure — templates where some fields are generated and others are constrained to a fixed option set or pattern (e.g. Microsoft's `guidance`), so the output conforms to a schema such as valid JSON.

**Intuition.** Rather than hoping the model emits well-formed structured output, force the format: let it generate the free-text fields while restricting others to valid options/regex, guaranteeing parseable results.

**Notes.** Useful for reliable JSON/structured output from an otherwise free-form generator. → In-context learning (zero/one/few-shot).

### Parameter-efficient fine-tuning (LoRA / PEFT)

**Definition.** Fine-tuning that updates only a small number of added parameters while freezing most of the pretrained weights. *LoRA* injects low-rank trainable matrices into the layers; *PEFT* is the general family.

**Intuition.** Full fine-tuning of a huge model is expensive; training a small set of adapter parameters (plus tricks like low-precision inference, e.g. bitsandbytes) makes customizing open base models like LLaMA far more accessible, at a fraction of the cost.

**Notes.** SFT-style fine-tuning is now achievable this way; full RLHF remains research-heavy. → In-context learning (zero/one/few-shot), Supervised fine-tuning (SFT).

### LLM limitations and safe use

**Definition.** Known failure modes of LLMs: they may be *biased*, *hallucinate* (fabricate facts), make *reasoning errors*, struggle with some task classes (e.g. spelling, a side effect of tokenization), have *knowledge cutoffs*, and be vulnerable to *prompt injection*, jailbreaks, and data-poisoning attacks.

**Intuition.** An LLM imitates the next token; it doesn't know what it doesn't know, doesn't sanity-check or reflect by default, and spends equal compute per token. This is *System 1* (fast, automatic) thinking — techniques like chain-of-thought, self-consistency, and tree-of-thoughts bolt on a *System 2* (slow, deliberate) layer. Its large fact-based knowledge and near-perfect context-window working memory are the real strengths to lean on.

**Notes.** Practical guidance: use in low-stakes settings with human oversight, treat outputs as inspiration/suggestions, and prefer copilots over autonomous agents. → Chain-of-thought prompting, Retrieval-augmented generation (RAG), Mode collapse.

### torch.compile

**Definition.** A PyTorch feature (`torch.compile(model)`) that traces the model's operations and compiles them into optimized, fused kernels ahead of execution.

**Intuition.** Eager PyTorch dispatches each operation separately with Python overhead; compiling fuses operations and strips that overhead, so the model runs substantially faster. The cost is a one-time compilation delay on the first call.

**Notes.** A near-free speedup for training and inference once compiled. *(Karpathy)* → Backpropagation, Model compression (distillation, pruning, quantization).

---

## Unsupervised Learning *(ISL ch. 12)*

### Unsupervised learning

**Definition.** Learning from predictors `X_1, …, X_p` with *no* response `Y` — the goal is to discover structure (patterns, groupings, low-dimensional summaries) rather than predict a labeled output. Main tools: PCA and clustering.

**Intuition.** No answer key, so you can't measure predictive error. Instead you explore — "what natural structure lives in this data?" Often part of exploratory data analysis.

**Notes.** Harder to validate than supervised learning: no test-set error or cross-validation to lean on, and no universally agreed way to check results. → Principal components analysis, Clustering, Supervised vs unsupervised learning (stat).

### Principal components analysis (PCA)

**Definition.** An unsupervised method that finds a few directions (*principal components*) capturing most of the variance in the data. The first principal component is the normalized linear combination `Z_1 = φ_{11}X_1 + … + φ_{p1}X_p` (with `Σ φ_{j1}² = 1`) having the largest variance; each later component has the largest remaining variance subject to being uncorrelated with (perpendicular to) the earlier ones. The weights `φ` are the *loadings*.

**Intuition.** Find the single direction the data spread out along most — that's PC1. Then the next-biggest spread at right angles to it, and so on. A few components often capture most of the variation, so you can compress or visualize high-dimensional data with little loss. The components also trace the line/plane closest to the data cloud.

**Notes.** Center variables to mean zero first, and usually scale them too — PCA is sensitive to units. An `n × p` dataset has `min(n−1, p)` components, but you keep only the first few. Feeding components into a regression is PCR. → Principal components regression (PCR) (stat), Unsupervised learning.

### Matrix completion

**Definition.** Filling in missing entries of a data matrix by approximating it with a low-rank (PCA-style) decomposition fit only on the *observed* entries, then reading off predictions for the missing ones. Solved by an iterative algorithm.

**Intuition.** If the data really live near a few underlying dimensions, the observed entries pin down that structure well enough to guess the gaps — the principle behind recommender systems ("users like you also liked…").

**Notes.** An application of the PCA idea to missing data. → Principal components analysis.

### Clustering

**Definition.** A broad family of methods for partitioning observations into subgroups (*clusters*) so that observations within a cluster are similar and those in different clusters are dissimilar.

**Intuition.** Find the natural groupings in data — customer segments, cell types — without being told the groups in advance.

**Notes.** Requires a notion of similarity/dissimilarity (often Euclidean distance), whose choice strongly affects results. Two main approaches: K-means and hierarchical. → K-means clustering, Hierarchical clustering.

### K-means clustering

**Definition.** Partitions data into a *pre-specified* number `K` of non-overlapping clusters. Algorithm: randomly assign points to clusters, then iterate — (a) compute each cluster's *centroid* (the mean of its points), (b) reassign each point to the nearest centroid — until assignments stop changing.

**Intuition.** Guess `K` groups, find their centers, move each point to its closest center, recompute centers, repeat. It settles into clusters that minimize within-cluster spread.

**Notes.** You must choose `K` up front, and the result depends on the random start — so run it several times and keep the best. → Clustering, Hierarchical clustering.

### Hierarchical clustering

**Definition.** Builds a tree of nested clusters *without* pre-specifying `K`. The *agglomerative* (bottom-up) version starts with each observation as its own cluster, then repeatedly fuses the two most similar clusters until all are joined. The result is a *dendrogram* (an upside-down tree) whose fusion heights show dissimilarity.

**Intuition.** Repeatedly merge the closest pair of groups, recording when each merge happens. Cutting the dendrogram at a chosen height gives any number of clusters — so you read `K` off the tree instead of committing to it in advance.

**Notes.** Fusing *groups* (not just points) needs *linkage* — a rule for cluster-to-cluster dissimilarity: complete, average, single, or centroid. Linkage and the distance measure both strongly shape the dendrogram. → Clustering, K-means clustering.

### Gaussian mixture model (GMM)

**Definition.** A probabilistic clustering model assuming the data were generated from a mixture of `k` Gaussian distributions with unknown parameters. The generative story: for each instance, pick cluster `j` at random with probability given by that cluster's weight `φ_j`; then draw the instance from that cluster's Gaussian, `N(μ_j, Σ_j)`. Instances from one Gaussian form a cluster.

**Intuition.** K-means assigns each point to its nearest centroid, full stop. A GMM instead models each cluster as an ellipsoidal blob with its own shape, size, density, and orientation, and gives each point a *probability* of belonging to each cluster. Softer and more flexible than K-means, which effectively assumes round, equal-sized clusters.

**Notes.** In the simplest variant you must specify `k` in advance. Fitting the parameters (weights, means, covariances) is done by the EM algorithm — *(beyond both books' highlights here)*. Related to QDA, which also models each class as its own Gaussian, but supervised. *(Hands-On ML)* → Clustering, K-means clustering, Quadratic discriminant analysis (QDA) (stat).

### Bayesian Gaussian mixture

**Definition.** A GMM variant that can drive the weights of unnecessary clusters to (or near) zero. Set the number of components to a value you have reason to believe exceeds the true number, and the algorithm prunes the surplus itself.

**Intuition.** Instead of hunting for the right `k` by hand, over-provision and let the fit decide: ask for 10 clusters on 3-cluster data and the extra weights collapse to zero, leaving three non-trivial clusters.

**Notes.** Needs some minimal prior knowledge — enough to pick a safe upper bound on `k`. A useful answer to the "how do I choose `K`?" problem that K-means leaves open. *(Hands-On ML)* → Gaussian mixture model, K-means clustering.

---

## Machine Learning Systems Design *(Designing ML Systems)*

### When to use machine learning

**Definition.** Machine learning is an approach to *learn* *complex patterns* from *existing data* and use them to make *predictions* on *unseen data*. It fits when all of these hold: there are patterns to learn, the patterns are complex (hard to hand-code), data exists (or can be collected), the problem is predictive, and the patterns generalize — ideally the task is also repetitive, operates at scale, and lives in a changing environment.

**Intuition.** With no pattern, no data, or a problem cheaply solved by explicit rules, ML is the wrong tool. It earns its keep when rules would be too many or too brittle, and when being right *most* of the time is already valuable.

**Notes.** Complements the statistical-learning framing (`Y = f(X) + ε`) by stressing the *systems* preconditions and deployment realities. → Statistical learning (stat), Prediction vs inference (stat).

### Classification task types (binary, multiclass, multilabel)

**Definition.** Within classification: *binary* (two classes), *multiclass* (one label from `K > 2` mutually exclusive classes), and *multilabel* (each example may carry *several* labels at once).

**Intuition.** Multiclass assumes exactly one class is correct (probabilities sum to 1, softmax); multilabel drops that assumption — an image can be both "cat" and "outdoors" — so it's usually framed as `K` independent binary problems (a sigmoid per label). Multilabel is the hardest to source labels for and to evaluate.

**Notes.** Very high-cardinality multiclass is sometimes decomposed hierarchically. → Regression vs classification (stat), Softmax, Multinomial logistic regression (stat), Classification metrics (precision, recall, F1).

### Decoupling objectives (multi-objective optimization)

**Definition.** When a model must satisfy several competing objectives (e.g. rank posts by *engagement* and by *quality*), train a separate model or head per objective and combine their outputs with a weighted (linear) sum, rather than optimizing one pre-blended loss.

**Intuition.** Folding objectives into a single loss `α·loss_a + β·loss_b` means retraining every time you re-weight the trade-off. Decoupling — optimize each objective separately, then combine scores with tunable weights at serving time — lets you re-weight without retraining and keeps each objective's behavior legible.

**Notes.** The combination weights become product knobs. → Neural network regularization.

### ETL (extract, transform, load)

**Definition.** The standard data pipeline: *extract* from sources, *transform* (clean, validate, aggregate, reshape), and *load* into a destination (warehouse, database, files) for downstream use.

**Intuition.** The plumbing that turns messy source data into query-ready tables feeding training and features. ("ELT" reorders it — load raw first, transform later inside the warehouse.)

**Notes.** → Batch features vs streaming features.

### Non-probability sampling

**Definition.** Selecting data by convenience rather than random chance: *convenience* (whatever's easily available), *snowball* (existing samples recruit more), *judgment* (an expert hand-picks), and *quota* (fill fixed counts per group).

**Intuition.** Cheap and common, but the selection criteria are non-random, so the sample carries selection bias and rarely represents the population — a frequent hidden source of biased models.

**Notes.** Fine for exploration; risky as the basis for a production model. → Random sampling methods (simple, stratified, weighted, reservoir), Data leakage.

### Random sampling methods (simple, stratified, weighted, reservoir)

**Definition.** Probability-based selection: *simple random* (every item equally likely), *stratified* (split into groups/strata and sample each so all are represented), *weighted* (each item drawn with a chosen probability), and *reservoir* (a streaming algorithm keeping a uniform sample of `k` items from a stream of unknown length).

**Intuition.** Stratified sampling guarantees rare classes appear; weighted sampling encodes priorities or corrects imbalance. Reservoir sampling solves the streaming case: keep the first `k`, then for the `i`-th arrival (`i > k`) keep it with probability `k/i`, evicting a random current member — leaving every item seen so far equally likely to be in the reservoir at any moment.

**Notes.** → Non-probability sampling, Class imbalance.

### Labeling and weak supervision

**Definition.** Producing labels without (or with little) hand-annotation. *Weak supervision* generates noisy labels programmatically via *labeling functions* — heuristics such as regex/keyword rules, existing knowledge bases, or the outputs of other models — which are then combined.

**Intuition.** Hand-labeling is slow and expensive; labeling functions encode cheap, noisy expert intuition at scale, and aggregating many weak signals yields usable (if imperfect) labels. Neighboring paradigms lower labeling cost differently: semi-supervised learning leverages unlabeled data, transfer learning reuses another task's model, active learning labels only the most informative points.

**Notes.** → Semi-supervised learning (stat), Transfer learning, Active learning.

### Transfer learning

**Definition.** Reusing a model trained on one (data-rich) task as the starting point for another related task — as a fixed feature extractor, or by fine-tuning it on the new task's smaller dataset.

**Intuition.** Representations learned on a big task (a pretrained language or vision model) capture general structure that transfers, so the target task needs far less data and compute. This is the foundation of the pretrain-then-adapt recipe.

**Notes.** → Pretraining and base models, Parameter-efficient fine-tuning (LoRA / PEFT), Zero-shot learning.

### Zero-shot learning

**Definition.** Making correct predictions for a task the system was *never trained on*, by leveraging training on a *related* task (also called zero-data learning).

**Intuition.** The system needs *no* data for the task at hand — but it still had to learn from data for a related task, so "zero-shot" does not mean "zero data" overall. Common with large pretrained models prompted to do new tasks.

**Notes.** → In-context learning (zero/one/few-shot), Transfer learning.

### Active learning

**Definition.** A labeling strategy in which the model chooses which unlabeled examples it would most benefit from having labeled next (e.g. the ones it is most uncertain about), and a human labels those.

**Intuition.** Not all labels are equally useful; labeling the most informative/uncertain points buys more accuracy per label than random labeling, cutting annotation cost.

**Notes.** → Labeling and weak supervision.

### Class imbalance

**Definition.** When some classes vastly outnumber others (e.g. fraud detection), biasing a model toward the majority. Addressed by *resampling* — undersample the majority and/or oversample the minority (e.g. SMOTE, which synthesizes new minority examples) — and by adjusting the loss (class weights, focal loss).

**Intuition.** Accuracy is useless under imbalance (just predict the majority). Rebalancing the data or reweighting the loss forces the model to care about the rare class; the choice of *metric* matters as much as the fix.

**Notes.** → Random sampling methods (simple, stratified, weighted, reservoir), Classification metrics (precision, recall, F1), Data augmentation.

### Handling missing data

**Definition.** Two broad strategies: *deletion* (drop rows or columns with missing values) and *imputation* (fill them — with the mean/median/mode, a constant, or a model's prediction).

**Intuition.** Deletion is simple but discards data and can bias the set when missingness isn't random. Imputation keeps the rows but injects assumptions; whether missingness is random (MCAR/MAR) or informative (MNAR) determines what's safe. The *fact* of missingness can itself be a useful feature.

**Notes.** Fit imputation statistics on the training set only. → Feature scaling, Data leakage.

### Categorical feature encoding (hashing trick)

**Definition.** Turning categories into numbers a model can use. Beyond one-hot/dummy encoding, the *hashing trick* maps each category through a hash function into a fixed number of buckets, handling unseen and high-cardinality categories in bounded space.

**Intuition.** One-hot explodes with high cardinality and can't represent categories unseen at training time. Hashing fixes the dimension in advance and absorbs new categories gracefully (at the cost of occasional collisions), which is why it suits large-scale, streaming category spaces.

**Notes.** → One-hot encoding, Qualitative predictors and dummy variables (stat), Embedding layer.

### Feature crossing

**Definition.** Creating a new feature by combining two or more existing features (e.g. `area = length × width`, or the pair `(marital_status, num_children)`), so a model can capture their joint effect.

**Intuition.** The same idea as interaction terms in linear models: crossing lets a model that treats features additively express effects that depend on *combinations*. Especially useful for linear models, and it helps DNNs learn faster — at the cost of more features and higher overfitting risk.

**Notes.** → Interaction terms (stat), Feature scaling.

### Data leakage

**Definition.** When information unavailable at prediction time leaks into training, inflating offline metrics and then collapsing in production. *Group leakage* is a common form: correlated or duplicate records split across train and test (e.g. multiple scans of one patient), so the test set isn't truly unseen.

**Intuition.** Leakage lets the model "cheat" — via a target-derived feature, scaling statistics computed over the whole dataset, time-travel in a time-series split, or grouped duplicates straddling the split. Suspiciously high performance, or a single feature with implausibly high correlation to the target, is a red flag.

**Notes.** Guard by splitting *before* any preprocessing, splitting by group/time where relevant, and checking feature–target correlations. → Train / dev / test split, Handling missing data, Feature scaling.

### Classification metrics (precision, recall, F1)

**Definition.** Beyond raw accuracy: *precision* = TP/(TP+FP) (of predicted positives, how many are correct), *recall* = TP/(TP+FN) (of actual positives, how many were caught), and *F1* = their harmonic mean `2·P·R/(P+R)`.

**Intuition.** Accuracy hides *which* errors you make and misleads under class imbalance. Precision vs recall is the false-positive vs false-negative trade-off; F1 balances them in one number. Pick the metric matching each error's cost — recall for disease screening, precision when false alarms are expensive.

**Notes.** All read off the confusion matrix; sweeping the threshold traces the ROC (or precision–recall) curve. → Confusion matrix and error types (stat), ROC curve (stat), Class imbalance.

### Stacking (stacked generalization)

**Definition.** An ensemble that trains several diverse *base* models and then a *meta-model* which learns how best to combine their predictions, instead of a fixed average or vote.

**Intuition.** Where bagging averages and boosting adds sequentially, stacking *learns* the combination — the meta-model discovers which base learner to trust when. Often squeezes out extra accuracy in competitions.

**Notes.** Train the meta-model on *out-of-fold* base predictions to avoid leakage. → Ensemble methods and weak learners, Bagging, Boosting.

### Distributed training (data vs model parallelism)

**Definition.** Training across multiple devices. *Data parallelism*: replicate the model on each device, split the batch across them, and average gradients. *Model parallelism*: split the *model itself* across devices (different layers/tensors on different GPUs) when it's too big to fit on one.

**Intuition.** Data parallelism scales throughput (more data per step) and is the common case; model parallelism is for models too large for one device's memory. Real systems often combine both, plus pipeline parallelism.

**Notes.** Data-parallel gradient averaging effectively enlarges the batch, which interacts with the learning rate. → Stochastic gradient descent (SGD), Batch normalization.

### Model evaluation tests (perturbation, invariance, slice-based)

**Definition.** Behavioral tests beyond a single aggregate score. *Perturbation test*: add small noise/changes to inputs and check the outputs stay sensible/robust. *Invariance test*: change inputs that *shouldn't* affect the output (e.g. a name or gender) and check the prediction doesn't move. *Slice-based evaluation*: measure performance separately on meaningful *subsets* (slices) — by segment, device, language — to expose failures the aggregate hides.

**Intuition.** A good overall number can mask that the model fails a critical subgroup or reacts to irrelevant features. These tests probe the robustness and fairness a lump-sum metric can't.

**Notes.** *Correction:* slice-based evaluation is **not** the train/validation/test split. The split partitions data to *estimate generalization*; slicing evaluates the *already-trained* model on subgroups of the *same* eval set to find hidden weaknesses — a different purpose. → Train / dev / test split, Classification metrics (precision, recall, F1).

### Batch features vs streaming features

**Definition.** *Batch (static) features* are precomputed from historical data at rest (e.g. a user's average rating over the last month), stored, and looked up. *Streaming (dynamic/online) features* are computed in near-real-time from a stream of events at request time (e.g. clicks in the last 10 minutes).

**Intuition.** Batch features are cheap and simple but can be stale; streaming features are fresh and often more predictive but need streaming infrastructure. Many systems use both.

**Notes.** *Clarification:* "streaming features" and "online features" usually mean the *same* thing — the real contrast is **batch/static vs streaming/online**, not "streaming vs online." → ETL, Batch prediction vs online prediction, Storage vs compute.

### Batch prediction vs online prediction

**Definition.** *Batch prediction* (asynchronous): precompute predictions for many inputs on a schedule and store them for lookup. *Online prediction* (on-demand/synchronous): generate a prediction the moment a request arrives.

**Intuition.** Batch is simpler and cheaper and hides model latency, but predictions can be stale and you must know the inputs in advance. Online serves fresh, input-dependent predictions (needed for search or recommendations on new queries) but demands low-latency serving. Batch features often feed batch prediction; streaming features enable responsive online prediction.

**Notes.** → Batch features vs streaming features, Model compression (distillation, pruning, quantization), Storage vs compute.

### Model compression (distillation, pruning, quantization)

**Definition.** Shrinking a model for cheaper/faster inference. *Knowledge distillation*: train a small *student* to mimic a large *teacher*'s outputs. *Pruning*: remove unimportant weights/neurons from a trained network (often zeroing small-magnitude weights). *Quantization*: store and compute weights at lower precision (e.g. 32-bit → 8-bit or lower).

**Intuition.** Big models are accurate but expensive to serve; these techniques trade a little accuracy for large gains in size, latency, and energy. Quantization is the most broadly used and the most hardware-friendly.

**Notes.** *Clarification:* "pruning" here is **neural-network** pruning — removing weights — distinct from decision-tree *cost-complexity pruning* (trimming branches), though both simplify a trained model. → Tree pruning (cost-complexity), torch.compile, Batch prediction vs online prediction.

### Data distribution shifts (covariate, label, concept)

**Definition.** When the data a deployed model sees drifts from its training distribution. *Covariate shift*: input distribution `P(X)` changes while `P(Y|X)` holds. *Label shift*: label distribution `P(Y)` changes while `P(X|Y)` holds. *Concept drift*: the relationship `P(Y|X)` itself changes (same inputs, different correct answer).

**Intuition.** A model trained on last year's data goes stale — slowly or suddenly — as the world changes. Identifying which distribution moved guides the fix: reweighting for covariate/label shift, retraining for concept drift.

**Notes.** Detected by monitoring input/output/label distributions over time. → Continual learning (stateless vs stateful), Degenerate feedback loops, Train / dev / test split.

### Degenerate feedback loops

**Definition.** When a model's own predictions shape the future data it is trained on, reinforcing its biases over time — e.g. a recommender that only surfaces popular items makes them more popular, so they look even better next round.

**Intuition.** The model shapes the world it later learns from, creating a self-fulfilling loop that narrows exposure and entrenches whatever it already favored (popularity bias, filter bubbles). Breaking it needs deliberate exploration (randomization, bandits) or positional debiasing.

**Notes.** → Data distribution shifts (covariate, label, concept), Bandit algorithms.

### Continual learning (stateless vs stateful)

**Definition.** How a model is updated over time. *Stateless retraining*: retrain from scratch on a fresh window of data each cycle. *Stateful training* (incremental/continual): continue training the *existing* model on new data, carrying its learned state forward.

**Intuition.** Stateless is simple and reproducible but wasteful and slower to adapt; stateful updates cheaply on just the new data and reacts faster — at the risk of *catastrophic forgetting* and harder debugging. Update cadence trades freshness against stability.

**Notes.** → Data distribution shifts (covariate, label, concept), Transfer learning.

### Deployment strategies (shadow, canary, A/B testing)

**Definition.** Ways to roll out and evaluate a new model in production. *Shadow deployment*: run the new model alongside the old on live traffic but don't serve its predictions — just log and compare. *Canary*: release to a small slice of traffic first, then widen. *A/B testing*: route a fraction of real traffic to the new model and compare outcomes statistically before full rollout.

**Intuition.** Offline metrics don't guarantee online wins. Shadow mode de-risks by testing on real inputs with zero user impact; A/B testing measures the real causal effect on user-facing metrics on a slice of traffic.

**Notes.** → Model evaluation tests (perturbation, invariance, slice-based), Bandit algorithms, Hypothesis test, t-statistic, and p-value (stat).

### Bandit algorithms

**Definition.** Online decision algorithms that balance *exploration* (trying options to learn their payoff) against *exploitation* (choosing the current best), allocating traffic adaptively — e.g. among candidate models or content. *Contextual bandits* condition the choice on features of each request.

**Intuition.** Plain A/B testing sends fixed traffic to a losing variant for the whole test; a bandit shifts traffic toward better performers as evidence accumulates, so it's more data-efficient. The exploration keeps it from prematurely locking onto a wrong "best."

**Notes.** A remedy for degenerate feedback loops (deliberate exploration). → Deployment strategies (shadow, canary, A/B testing), Degenerate feedback loops.

### Storage vs compute

**Definition.** A recurring systems trade-off: precompute-and-store results (spend storage) versus compute-on-demand (spend compute) — as in batch vs streaming features, or batch vs online prediction.

**Intuition.** Storing precomputed answers makes serving cheap and fast but risks staleness and grows storage; computing fresh each time is always current but costs latency and compute per request. The right balance depends on freshness needs, request volume, and cost.

**Notes.** → Batch features vs streaming features, Batch prediction vs online prediction.

---

## Controlled Experiments and A/B Testing *(Trustworthy Online Controlled Experiments)*

### A/B testing (controlled experiments)

**Definition.** A **controlled experiment** randomly splits users into a **Control** (A, the current experience) and one or more **Treatments** (B, …, the change), then compares a metric between them. Because assignment is random, a difference in the metric can be attributed **causally** to the change.

**Intuition.** Randomization makes the groups statistically equivalent on everything *except* the change, so any outcome gap is caused by the change — not by a confound. It's the gold standard for "does this actually help?" Larger designs (A/B/C/D) compare several options at once.

**Example.** Ship a new checkout button to 50% of users (Treatment) and keep the old one for 50% (Control); if Treatment's purchase rate beats Control's beyond noise (a two-sample test, → (stat)), the button *caused* the lift.

**Notes.** The online analogue of a randomized controlled trial. → Deployment strategies (shadow, canary, A/B testing), Bandit algorithms, Overall Evaluation Criterion (OEC), Hypothesis test, t-statistic, and p-value (stat).

### Overall Evaluation Criterion (OEC)

**Definition.** The **OEC** is the single agreed metric (or weighted combination) an experiment optimizes — chosen to reflect **long-term** value, not just a short-term bump.

**Intuition.** You get what you measure. A bad OEC (clicks alone) is easily gamed by changes that hurt the business — clickbait raises clicks but lowers trust and retention. A good OEC balances short-term signal against long-term goals, so "winning" the test means genuinely improving.

**Example.** Optimizing *sessions per user* or *long-term revenue* rather than immediate clicks discourages dark-pattern changes that spike clicks but drive users away.

**Notes.** The hardest and most important design choice in experimentation. → Metric taxonomy (goal, driver, guardrail), Decoupling objectives (multi-objective optimization).

### Parameters, variants, and randomization unit

**Definition.** A **parameter** (factor/variable) is a controllable knob thought to affect the OEC, set to different **levels** (values). A **variant** is one user experience (a setting of the parameters) — A and B are the Control and Treatment variants. The **randomization unit** is what you randomly assign (usually a user, via hashing) to a variant. A **multivariate test (MVT)** varies several parameters together to find interactions.

**Intuition.** Hashing the unit (e.g. user id) to a variant gives a stable, pseudo-random split so the same user always sees the same variant and the groups stay comparable. Randomizing by *user* (not by page view) keeps a person's experience consistent and avoids correlated observations.

**Example.** Parameter = button colour with levels {blue, green} → two variants. An MVT crossing colour × font size finds the best *combination*, catching interactions a one-parameter test would miss.

**Notes.** Proper randomization is exactly what licenses the causal claim. → A/B testing (controlled experiments), Interaction terms (stat).

### Metric taxonomy (goal, driver, guardrail)

**Definition.** Three roles a metric can play. **Goal** (success / true-north) metrics capture what the org ultimately wants. **Driver** (sign-post / surrogate / predictive) metrics are shorter-term, more sensitive proxies believed to *lead* to the goal. **Guardrail** metrics guard against harm — some protect the business, others protect the *trustworthiness / internal validity* of the experiment itself.

**Intuition.** Goal metrics are often slow and hard to move in one experiment, so you steer by driver metrics that respond faster, while guardrails make sure a "win" didn't break something (latency, crashes, revenue) or invalidate the test (e.g. a sample-ratio mismatch).

**Example.** Goal = customer lifetime value; driver = weekly active days; guardrails = page-load time and error rate (business) plus the traffic split actually matching the intended 50/50 (trustworthiness).

**Notes.** → Overall Evaluation Criterion (OEC), Good metric properties (measurable, attributable, sensitive).

### Good metric properties (measurable, attributable, sensitive)

**Definition.** An experiment metric should be **measurable** (you can actually quantify it — some effects, like post-purchase satisfaction, are hard), **attributable** (each value can be tied to a specific variant), and **sensitive & timely** (able to detect changes that matter, soon enough to act on).

**Intuition.** A metric you can't attribute to a variant can't be compared across A and B; a metric too insensitive won't budge even when the change truly helps, wasting the experiment. These properties decide whether a metric is *usable* at all — before you even ask whether it's the right thing to optimize.

**Example.** "Revenue per user" is measurable, attributable, and fairly sensitive; "brand love" matters but is hard to measure and attribute, so it makes a poor *experiment* metric.

**Notes.** Sensitivity ties directly to statistical power — an insensitive metric needs a huge sample to move. → Statistical power and power analysis (stat), Metric taxonomy (goal, driver, guardrail).

### Twyman's law and trustworthiness

**Definition.** **Twyman's law:** any figure that looks especially interesting or surprising is probably *wrong* — a data or instrumentation error — and should be double-checked before it's believed.

**Intuition.** Spectacular results (a 50% lift!) far more often come from bugs, logging errors, or broken randomization than from genuinely huge effects. Trustworthy experimentation means distrusting the too-good result and validating it (guardrails, A/A tests, sanity checks) before acting.

**Example.** A treatment showing an implausible 30% revenue jump usually traces to a tracking bug or a sample-ratio mismatch, not a real win — investigate before shipping.

**Notes.** The cultural backbone of a trustworthy experimentation platform. → Metric taxonomy (goal, driver, guardrail), Model evaluation tests (perturbation, invariance, slice-based).

---

## Glossary

- **A/B test (controlled experiment)** — randomized Control vs Treatment comparison for causal effects. → [A/B testing (controlled experiments)](#ab-testing-controlled-experiments).
- **Activation function** — the non-linearity (sigmoid, ReLU) inside a neural-net unit. → [Activation function](#activation-function).
- **Active learning** — the model requests labels for its most informative examples. → [Active learning](#active-learning).
- **Adam / AdamW** — adaptive per-parameter optimizer; AdamW decouples weight decay. → [Adam and AdamW](#adam-and-adamw).
- **Adding vocabulary after training** — extend embedding/output rows and train only those, frozen elsewhere. → [Adding vocabulary after training](#adding-vocabulary-after-training).
- **Autocorrelation / autoregressive (AR) model** — time-series values correlate with their own lags; AR(L) regresses on the previous `L` values. → [Autoregressive models and autocorrelation](#autoregressive-models-and-autocorrelation).
- **Bag-of-words** — represent a document by which dictionary words it contains. → [Bag-of-words](#bag-of-words).
- **Activation saturation** — squashing units pinned at ±1 (or dead at 0) with ~0 gradient. → [Activation saturation](#activation-saturation).
- **Backpropagation** — forward pass, chain-rule backward pass, then a gradient step. → [Backpropagation](#backpropagation).
- **Bandit algorithms** — adaptively balance exploration vs exploitation over traffic. → [Bandit algorithms](#bandit-algorithms).
- **Base model** — pretraining-only next-token predictor; not yet an assistant. → [Pretraining and base models](#pretraining-and-base-models).
- **Batch vs online prediction** — precomputed/stored vs generated on-demand at request time. → [Batch prediction vs online prediction](#batch-prediction-vs-online-prediction).
- **Batch vs streaming features** — precomputed static vs near-real-time online features. → [Batch features vs streaming features](#batch-features-vs-streaming-features).
- **Broadcasting** — stretch a smaller tensor to match a larger one for element-wise ops; mind `keepdims`. → [Broadcasting](#broadcasting).
- **Bagging** — averaging trees fit on bootstrap samples to cut variance. → [Bagging](#bagging).
- **BART** — tree ensemble combining random perturbation with boosting-style residual fitting. → [Bayesian additive regression trees (BART)](#bayesian-additive-regression-trees-bart).
- **Bayesian Gaussian mixture** — GMM variant that zeroes out surplus clusters. → [Bayesian Gaussian mixture](#bayesian-gaussian-mixture).
- **Batch normalization** — normalize pre-activations across the batch; learned scale/shift; mild regularizer. → [Batch normalization](#batch-normalization).
- **Boosting** — sequential trees each fit to the previous model's residuals. → [Boosting](#boosting).
- **Byte-pair encoding (BPE)** — subword tokenization by merging frequent token pairs. → [Tokenization and byte-pair encoding (BPE)](#tokenization-and-byte-pair-encoding-bpe).
- **CatBoost / XGBoost / LightGBM** — production gradient-boosting libraries; CatBoost adds ordered boosting + native categoricals. → [Gradient boosting libraries (XGBoost, LightGBM, CatBoost)](#gradient-boosting-libraries-xgboost-lightgbm-catboost).
- **Causal (masked) self-attention** — attention restricted to current and past tokens via a triangular mask. → [Causal (masked) self-attention](#causal-masked-self-attention).
- **Chain-of-thought (CoT)** — prompt the model to reason in steps before answering. → [Chain-of-thought prompting](#chain-of-thought-prompting).
- **Classification task types** — binary, multiclass (one of K), multilabel (several at once). → [Classification task types (binary, multiclass, multilabel)](#classification-task-types-binary-multiclass-multilabel).
- **Class imbalance** — skewed class counts; fix via resampling (SMOTE) or loss weights. → [Class imbalance](#class-imbalance).
- **Clustering** — partitioning observations into similar subgroups. → [Clustering](#clustering).
- **Computational graph / autograd** — recorded op graph that makes derivatives automatic. → [Computational graph and autograd](#computational-graph-and-autograd).
- **Constrained prompting** — templates forcing structured (e.g. JSON) output. → [Constrained prompting](#constrained-prompting).
- **Continual learning** — stateless retraining vs stateful incremental training. → [Continual learning (stateless vs stateful)](#continual-learning-stateless-vs-stateful).
- **Convolution filter** — small learned matrix slid over image patches to detect local patterns. → [Convolution filter](#convolution-filter).
- **Convolutional neural network (CNN)** — image network of convolution + pooling layers. → [Convolutional neural network (CNN)](#convolutional-neural-network-cnn).
- **Cross-attention** — Q from one sequence, K/V from another (e.g. encoder → decoder). → [Encoder, decoder, and cross-attention](#encoder-decoder-and-cross-attention).
- **Cross-entropy** — the classification loss for neural nets (negative log-likelihood). → [Output layer and loss](#output-layer-and-loss).
- **Data augmentation** — label-preserving random distortions that enlarge training data. → [Data augmentation](#data-augmentation).
- **Data distribution shift** — covariate P(X), label P(Y), or concept P(Y|X) drift. → [Data distribution shifts (covariate, label, concept)](#data-distribution-shifts-covariate-label-concept).
- **Data leakage** — train-time use of info unavailable at prediction; includes group leakage. → [Data leakage](#data-leakage).
- **Decoder-only transformer (GPT)** — stacked masked-attention blocks; the GPT architecture. → [Decoder-only transformer (GPT)](#decoder-only-transformer-gpt).
- **Decoupling objectives** — train a head per objective, combine scores with tunable weights. → [Decoupling objectives (multi-objective optimization)](#decoupling-objectives-multi-objective-optimization).
- **Degenerate feedback loop** — predictions shape future training data, entrenching bias. → [Degenerate feedback loops](#degenerate-feedback-loops).
- **Dense (fully connected) layer** — every neuron connected to every neuron of the previous layer; `φ(XW + b)`. → [Dense (fully connected) layer](#dense-fully-connected-layer).
- **Decision tree** — flowchart of splits carving the predictor space into regions. → [Decision trees](#decision-trees).
- **Dendrogram** — the nested-cluster tree produced by hierarchical clustering. → [Hierarchical clustering](#hierarchical-clustering).
- **Distributed training** — data parallelism (split the batch) vs model parallelism (split the model). → [Distributed training (data vs model parallelism)](#distributed-training-data-vs-model-parallelism).
- **Double descent** — test error can fall again past the interpolation point. → [Double descent](#double-descent).
- **Dropout** — regularize a net by randomly zeroing units during training. → [Dropout](#dropout).
- **Early stopping** — halt training when validation performance stops improving. → [Early stopping](#early-stopping).
- **Encoder / decoder** — bidirectional vs masked transformer stacks; GPT is decoder-only. → [Encoder, decoder, and cross-attention](#encoder-decoder-and-cross-attention).
- **Ensemble / weak learner** — combine many mediocre models into a strong one. → [Ensemble methods and weak learners](#ensemble-methods-and-weak-learners).
- **ETL** — extract, transform, load: the data pipeline. → [ETL (extract, transform, load)](#etl-extract-transform-load).
- **Feature crossing** — combine features into a new joint feature (an interaction). → [Feature crossing](#feature-crossing).
- **Embedding layer** — learned lookup table mapping tokens to dense vectors. → [Embedding layer](#embedding-layer).
- **Feature scaling** — min-max scaling vs standardization to put attributes on one scale. → [Feature scaling](#feature-scaling).
- **Feed-forward network** — inputs flow through hidden layers to an output. → [Neural networks (feed-forward)](#neural-networks-feed-forward).
- **Gaussian mixture model (GMM)** — probabilistic clustering as a mixture of `k` Gaussians. → [Gaussian mixture model (GMM)](#gaussian-mixture-model-gmm).
- **Goal / driver / guardrail metric** — true-north / leading-proxy / protective experiment metrics. → [Metric taxonomy (goal, driver, guardrail)](#metric-taxonomy-goal-driver-guardrail).
- **Gradient descent** — fit by stepping parameters downhill on the loss. → [Gradient descent](#gradient-descent).
- **Hashing trick** — hash categories into fixed buckets for high-cardinality encoding. → [Categorical feature encoding (hashing trick)](#categorical-feature-encoding-hashing-trick).
- **Hinge loss** — `max(0, 1 − t)`; the SVM's margin objective. → [Hinge loss](#hinge-loss).
- **Hidden layer / units** — a neural net's layers of derived features. → [Hidden layers and units](#hidden-layers-and-units).
- **Hierarchical clustering** — merge clusters bottom-up into a dendrogram; no `K` needed. → [Hierarchical clustering](#hierarchical-clustering).
- **Hyperplane** — flat `(p−1)`-dimensional divider; basis of SVMs. → [Hyperplane](#hyperplane).
- **In-context learning (zero/one/few-shot)** — task behavior from prompt examples, no gradient updates. → [In-context learning (zero/one/few-shot)](#in-context-learning-zeroonefew-shot).
- **K-means clustering** — partition into `K` clusters via centroids. → [K-means clustering](#k-means-clustering).
- **Knowledge distillation** — train a small student model to mimic a large teacher. → [Model compression (distillation, pruning, quantization)](#model-compression-distillation-pruning-quantization).
- **Layer normalization** — normalize each token's own features (batch-independent); transformer default. → [Layer normalization](#layer-normalization).
- **Leaky ReLU / ELU / SELU** — ReLU variants; rough ordering SELU > ELU > leaky ReLU > ReLU > tanh > logistic. → [Activation function](#activation-function).
- **Language model (n-gram / bigram)** — predicts the next token from previous ones. → [Language model](#language-model).
- **Learning-rate schedule** — vary/decay the step size over training. → [Learning-rate schedule](#learning-rate-schedule).
- **Learning rate** — the step size in gradient descent. → [Gradient descent](#gradient-descent).
- **Linkage** — rule for cluster-to-cluster dissimilarity (complete/average/single/centroid). → [Hierarchical clustering](#hierarchical-clustering).
- **Likelihood / NLL** — product of assigned probabilities; negate the log for a loss. → [Likelihood and negative log-likelihood](#likelihood-and-negative-log-likelihood).
- **LLM agent (ReAct)** — an LLM in a reason–act–observe loop with tools. → [LLM agents (chains, ReAct)](#llm-agents-chains-react).
- **LLM limitations** — bias, hallucination, reasoning errors, cutoffs, prompt injection. → [LLM limitations and safe use](#llm-limitations-and-safe-use).
- **Loadings** — the predictor weights defining a principal component. → [Principal components analysis (PCA)](#principal-components-analysis-pca).
- **Logits** — un-normalized scores (log-counts) fed into softmax. → [Logits](#logits).
- **LoRA / PEFT** — parameter-efficient fine-tuning via a few added weights. → [Parameter-efficient fine-tuning (LoRA / PEFT)](#parameter-efficient-fine-tuning-lora--peft).
- **LSTM** — RNN variant with long- and short-term memory tracks. → [LSTM](#lstm).
- **Max-norm regularization** — cap each neuron's incoming weight norm at `r`. → [Max-norm regularization](#max-norm-regularization).
- **MC dropout** — keep dropout on at prediction time and average many passes. → [MC dropout](#mc-dropout).
- **Mercer's theorem** — licence for the kernel trick: a valid kernel implies a feature map exists. → [Mercer's theorem](#mercers-theorem).
- **Matrix completion** — fill missing entries via a low-rank PCA-style fit. → [Matrix completion](#matrix-completion).
- **Maximal margin classifier** — separating hyperplane with the widest buffer. → [Maximal margin classifier](#maximal-margin-classifier).
- **Missing data (imputation / deletion)** — drop or fill missing values. → [Handling missing data](#handling-missing-data).
- **Mode collapse** — RLHF's loss of output diversity/entropy. → [Mode collapse](#mode-collapse).
- **Model compression** — distillation, (network) pruning, quantization for cheaper serving. → [Model compression (distillation, pruning, quantization)](#model-compression-distillation-pruning-quantization).
- **Multi-head attention** — several parallel attention heads concatenated and projected. → [Multi-head attention](#multi-head-attention).
- **Multilabel classification** — each example may carry several labels; per-label sigmoids. → [Classification task types (binary, multiclass, multilabel)](#classification-task-types-binary-multiclass-multilabel).
- **Multi-layer perceptron (MLP)** — stacked perceptron layers; solves XOR. → [Multi-layer perceptron (MLP) and the XOR problem](#multi-layer-perceptron-mlp-and-the-xor-problem).
- **Neural network** — layered model of derived features; basis of deep learning. → [Neural networks (feed-forward)](#neural-networks-feed-forward).
- **Neural network regularization** — early stopping, L1/L2, dropout, max-norm. → [Neural network regularization](#neural-network-regularization).
- **Non-probability sampling** — convenience/snowball/judgment/quota; cheap but biased. → [Non-probability sampling](#non-probability-sampling).
- **Numerical stability (log-sum-exp)** — subtract the max logit before exp so nothing overflows. → [Numerical stability](#numerical-stability).
- **OEC (Overall Evaluation Criterion)** — the long-term metric an experiment optimizes. → [Overall Evaluation Criterion (OEC)](#overall-evaluation-criterion-oec).
- **One-hot encoding** — index → length-K vector, 1 in that slot. → [One-hot encoding](#one-hot-encoding).
- **Out-of-bag (OOB) error** — free test-error estimate from bagging's unused points. → [Out-of-bag (OOB) error](#out-of-bag-oob-error).
- **Perceptron / TLU** — weighted sum plus a step function; single layer of threshold units. → [Artificial neuron and the perceptron](#artificial-neuron-and-the-perceptron).
- **Perceptron learning rule** — error-driven weight update `w ← w + η(y − ŷ)x`. → [Perceptron learning rule](#perceptron-learning-rule).
- **Perturbation / invariance / slice tests** — robustness, irrelevant-input, and subgroup evaluation. → [Model evaluation tests (perturbation, invariance, slice-based)](#model-evaluation-tests-perturbation-invariance-slice-based).
- **Pooling** — downsampling (e.g. max pooling) in a CNN for compactness and location invariance. → [Pooling](#pooling).
- **Positional encoding** — inject token order into order-agnostic attention. → [Positional encoding](#positional-encoding).
- **Precision / recall / F1** — of-predicted-positive / of-actual-positive / their harmonic mean. → [Classification metrics (precision, recall, F1)](#classification-metrics-precision-recall-f1).
- **Pretraining** — next-token training on a huge corpus to make a base model. → [Pretraining and base models](#pretraining-and-base-models).
- **Principal components analysis (PCA)** — unsupervised top-variance directions of the data. → [Principal components analysis (PCA)](#principal-components-analysis-pca).
- **Pruning (cost-complexity)** — trim a grown tree via an `α·|T|` penalty. → [Tree pruning (cost-complexity)](#tree-pruning-cost-complexity).
- **Quantization** — store and compute weights at lower precision. → [Model compression (distillation, pruning, quantization)](#model-compression-distillation-pruning-quantization).
- **Randomization unit** — what is randomly assigned to a variant (usually a hashed user). → [Parameters, variants, and randomization unit](#parameters-variants-and-randomization-unit).
- **Regex pre-tokenization** — split text before BPE so merges don't cross word/space boundaries. → [Regex pre-tokenization](#regex-pre-tokenization).
- **Reservoir sampling** — uniform sample of k items from a stream of unknown length. → [Random sampling methods (simple, stratified, weighted, reservoir)](#random-sampling-methods-simple-stratified-weighted-reservoir).
- **Residual (skip) connection** — `x = x + sublayer(x)`; a gradient highway for deep nets. → [Residual (skip) connections](#residual-skip-connections).
- **Retrieval-augmented generation (RAG)** — fetch documents into the context window at query time. → [Retrieval-augmented generation (RAG)](#retrieval-augmented-generation-rag).
- **Reward model (RM) / RLHF** — score responses from human preferences, then RL-optimize against it. → [Reward modeling and RLHF](#reward-modeling-and-rlhf).
- **Random forests** — bagging + random feature subsets per split to decorrelate trees. → [Random forests](#random-forests).
- **Recurrent neural network (RNN)** — sequence network with a running hidden state. → [Recurrent neural network (RNN)](#recurrent-neural-network-rnn).
- **Regression tree** — tree predicting the mean response in each region. → [Regression trees](#regression-trees).
- **ReLU / sigmoid** — the two common activation functions. → [Activation function](#activation-function).
- **Scaled dot-product attention** — divide scores by √(head size) to keep softmax diffuse. → [Scaled dot-product attention](#scaled-dot-product-attention).
- **Self-attention** — tokens communicate via query/key/value weighted sums. → [Self-attention](#self-attention).
- **Self-consistency** — majority-vote over multiple sampled reasoning paths. → [Self-consistency and ensembling attempts](#self-consistency-and-ensembling-attempts).
- **Shadow / canary / A/B deployment** — log-only, small-slice, and split-traffic rollouts. → [Deployment strategies (shadow, canary, A/B testing)](#deployment-strategies-shadow-canary-ab-testing).
- **Softmax** — exponentiate logits then normalize into class probabilities. → [Softmax](#softmax).
- **Special tokens** — reserved non-content tokens (e.g. `<|endoftext|>`) added outside BPE. → [Special tokens](#special-tokens).
- **Stacking** — a meta-model learns to combine base models' predictions. → [Stacking (stacked generalization)](#stacking-stacked-generalization).
- **Stochastic gradient descent (SGD)** — gradient steps on random minibatches. → [Stochastic gradient descent (SGD)](#stochastic-gradient-descent-sgd).
- **Storage vs compute** — precompute-and-store vs compute-on-demand trade-off. → [Storage vs compute](#storage-vs-compute).
- **Stratified sampling** — sample within groups so all strata are represented. → [Random sampling methods (simple, stratified, weighted, reservoir)](#random-sampling-methods-simple-stratified-weighted-reservoir).
- **Supervised fine-tuning (SFT)** — imitate high-quality demonstration responses. → [Supervised fine-tuning (SFT)](#supervised-fine-tuning-sft).
- **Support vector classifier** — soft-margin linear classifier tolerating violations. → [Support vector classifier (soft margin)](#support-vector-classifier-soft-margin).
- **Support vector machine (SVM)** — kernel-enlarged classifier for non-linear boundaries. → [Support vector machine (kernels)](#support-vector-machine-kernels).
- **tiktoken / SentencePiece** — BPE on UTF-8 bytes vs on Unicode code points. → [tiktoken vs SentencePiece](#tiktoken-vs-sentencepiece).
- **Tokenization** — reversible text↔integer mapping (char, BPE, SentencePiece). → [Tokenization and byte-pair encoding (BPE)](#tokenization-and-byte-pair-encoding-bpe).
- **Tokenization failure modes** — spelling, arithmetic, non-English, glitch tokens trace to tokenization. → [Tokenization failure modes](#tokenization-failure-modes).
- **torch.compile** — compiles/fuses model ops for a large speedup after a one-time cost. → [torch.compile](#torchcompile).
- **Transfer learning** — reuse a model trained on a related, data-rich task. → [Transfer learning](#transfer-learning).
- **Transformer block** — attention + feed-forward, each with a residual and layer norm. → [Transformer block](#transformer-block).
- **Tree of thoughts** — search a tree of intermediate reasoning steps. → [Tree of thoughts](#tree-of-thoughts).
- **tanh** — S-shaped activation with output in `(−1, 1)`, centered near 0. → [Activation function](#activation-function).
- **Train / dev / test split** — fit / tune / final-check partitions. → [Train / dev / test split](#train--dev--test-split).
- **Twyman's law** — a surprising figure is probably an error; verify it. → [Twyman's law and trustworthiness](#twymans-law-and-trustworthiness).
- **Unsupervised learning** — finding structure with no response variable. → [Unsupervised learning](#unsupervised-learning).
- **Vanishing / exploding gradients** — gradients decaying or blowing up through deep layers. → [Vanishing and exploding gradients](#vanishing-and-exploding-gradients).
- **Variant / parameter / level** — a tested experience; a knob (factor) set to values. → [Parameters, variants, and randomization unit](#parameters-variants-and-randomization-unit).
- **WaveNet / hierarchical context** — fuse a sequence progressively in a tree. → [WaveNet / hierarchical context](#wavenet--hierarchical-context).
- **Weak supervision** — programmatic noisy labels from regex/heuristics/other models. → [Labeling and weak supervision](#labeling-and-weak-supervision).
- **Weight init (Kaiming / fan-in)** — scale weights by gain/√(fan_in) to keep activations healthy. → [Weight initialization](#weight-initialization).
- **Weight initialization** — must be random, to break symmetry between units. → [Weight initialization](#weight-initialization).
- **Word embeddings** — dense vectors placing similar words nearby (word2vec, GloVe). → [Embedding layer](#embedding-layer).
- **Zeroing gradients** — clear `.grad` before each backward pass; gradients accumulate. → [Zeroing gradients](#zeroing-gradients).
- **Zero-shot learning** — predict a task with no data for it, via a related task. → [Zero-shot learning](#zero-shot-learning).