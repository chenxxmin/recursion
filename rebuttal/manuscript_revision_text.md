# Manuscript revision text

Author note: these are prepared replacement/addition paragraphs. They have not been incorporated into the manuscript PDF; editable manuscript sources are not present in this repository. Headings below identify insertion points.

## Abstract / introduction: scope and research questions

We study how Transformers learn sequential modular recurrences under multiple rules and incomplete observations. Three questions guide the experiments: how task structure changes the benefits of depth and width; which prediction formulas models use when intermediate values are unavailable; and how the availability and delay of algebraic shortcuts affect learning across positions. The setting supplies exact targets and alternative formulas that can be tested through input interventions. Its motivation comes from sequential state updates and context-dependent operations in language processing; the empirical conclusions concern the recurrence tasks studied here.

For example, “Alice holds a red ball and Bob a blue ball; they swap; Alice gives her ball to Carol” requires updating object ownership to infer that Bob holds the red ball. Such examples motivate studying state updates and implicit intermediate information, without equating natural language with a finite-order linear recurrence. Inter-trajectory learning abstracts adaptation to persistent but implicit context, such as domain or style: when continuing an unlabelled technical passage, a model must infer and maintain its linguistic conventions. Intra-trajectory learning abstracts explicitly instructed changes of operation, such as summarizing and then translating: the instruction identifies the operation, which the model must execute using the accumulated context. These are complementary task organizations, and we do not infer an intrinsic difficulty ordering from their direct comparison.

## Section 2.1: definitions

Let $p$ be prime. Each sequence value $q_t$ and coefficient $a_i$ is an integer in $\{0,\ldots,p-1\}$, with all arithmetic performed modulo $p$. A rule is a coefficient vector $\mathbf a=(a_1,\ldots,a_k)$. Given initial values $q_1,\ldots,q_k$, it generates

$$q_t=\left(\sum_{i=1}^k a_iq_{t-i}\right)\bmod p,\qquad t=k+1,\ldots,T.$$

Thus $a_1$ multiplies the most recent value. For arguments ordered oldest to newest,

$$f_{\mathbf a}(x_1,\ldots,x_k)=\left(\sum_{i=1}^k a_ix_{k+1-i}\right)\bmod p.$$

The rule pool is $\mathcal A=\{\mathbf a^{(1)},\ldots,\mathbf a^{(N)}\}$. An intra-trajectory tag $c_t$ identifies the vector used at position $t$. Because the current rule is sampled independently, hiding its identity can leave the same observed prefix compatible with different next values. The tag makes this choice observable; it does not supply the numerical answer. For example, $p=7$, $\mathbf a=(1,1)$ and initial values $1,2$ give $1,2,3,5,1,6,\ldots$.

## Section 2.2: prediction and length

Training and the reported token-level learning curves use causal attention and teacher forcing. Each prediction conditions on the data-provided prefix, with masking applied where specified; model predictions are not fed back as later inputs. Loss is applied to $q_{k+1},\ldots,q_T$ for single-rule and intra-trajectory tasks, and $q_{k+2},\ldots,q_T$ for inter-trajectory tasks. Rule tags are conditioning inputs, not prediction targets. $T$ counts numerical positions: ordinarily $T=16$, giving 30 input tokens for order-two intra sequences with 14 tags. The main masking experiments use numerical training length 64; masking replaces values without increasing the number of positions. Training and evaluation lengths must be listed separately where they differ.

## Section 2.2 / Appendix B.1: data protocols

For enumerated order-two datasets, we allocate $\lfloor0.7p^k\rfloor$ initial states per rule to training and the remainder to testing. Whole orbits are allocated where possible, with at most one orbit divided into contiguous portions to satisfy the quota. Initial states are disjoint, but complete windows can cross the two interfaces of a split cycle; boundary windows were not purged. The representative clean $(1,1)$ baseline with $p=127,T=16$ has 28 shared local states and 26 shared complete transitions, corresponding to 182 of 67,746 test predictions. These counts hold for all three audited seeds and are not asserted for every configuration.

Higher-order experiments use a fixed training budget and fresh evaluation samples. Training initial states were selected by orbit traversal in some historical runs and uniformly at random in others; the selection method should be identified per experiment. Uniform initial-state selection avoids requiring the training budget to occupy contiguous orbit segments with repeated local transitions. Fresh evaluation does not itself guarantee disjoint local states. For tagged intra-trajectory tasks, the sequence space has size $p^kN^{T-k}$; we sample 300,000 examples rather than enumerate a fixed fraction of this space.

## End of Section 2.2: methods overview

To identify candidate prediction formulas, we use attention patterns to select historical positions, randomize observed values to break algebraic correlations, and fit modular linear formulas to model predictions while retaining the specified mask pattern. We report agreement between these formulas and model outputs. Local-rule tests instead alter early history while preserving a valid informative suffix for inter-trajectory tasks, or retain rule tags while randomizing values for intra-trajectory tasks. Linear probes measure whether rule identity can be decoded from frozen representations. These analyses respectively test formula agreement, dependence on trajectory consistency, and representation decodability; they do not by themselves recover a complete neural circuit. Details appear in Appendices D, G.2, and G.3.

## Opening of Section 3: capability roadmap

We vary rule count, missing observations, and recurrence order to characterize the tasks learned under each training protocol. Depth and width have different effects across these settings, while algebraic shortcuts and their delays help explain differences in learning across positions. Section 4 examines how curriculum training changes the observed learning behavior; Section 5 studies interactions between rules, rule representations, and prediction strategies. The architecture results concern the tested configurations and budgets, rather than universal impossibility claims or equal-compute scaling laws.

## Section 3.3: algebraic shortcut clarification

An algebraic shortcut is an alternative linear prediction formula using fewer historical operands; it may require a larger maximum lag. For example, subtracting consecutive instances of $q_t=q_{t-1}+q_{t-2}+q_{t-3}$ gives $q_t=2q_{t-1}-q_{t-4}$ modulo $p$. This formula applies from $q_5$ but cannot predict the supervised target $q_4$. The distinction concerns the formula's information requirements and observed learning dynamics, not an inherent preference for one algebraically valid computation. Loss weighting tests how emphasizing early positions affects learning; a strategy combining formulas can also solve the task. Appendix D.3 reports models whose recovered formulas match the original recurrence at early positions and a sparse shortcut once sufficient history is available.

## Section 5.3: why distinguish prediction strategies?

Strategies that agree on valid trajectories can differ in their sensitivity to earlier context. We therefore test whether high prediction accuracy is accompanied by reliance on redundant history: while preserving sufficient information for local prediction, we perturb the earlier context and assess agreement with the current rule. This breaks global trajectory consistency and tests whether prediction remains possible from the information retained. In inter-trajectory tasks, the valid recent suffix must also provide rule information; in intra-trajectory tasks, the tag identifies the rule. These interventions characterize robustness to the tested contextual perturbations, without presupposing that one strategy is universally preferable.

## Conclusion: specific contribution and scope

The experiments characterize task-dependent architectural responses in multi-rule modular recurrence learning, identify alternative prediction formulas under missing observations, and relate delayed shortcuts to learning across positions. Together, the results provide an experimentally tractable setting for testing hypotheses about Transformer prediction strategies. The nonlinear experiments in Appendix F broaden the rule family within modular arithmetic; neither these experiments nor the language examples establish that the same findings hold for natural-language tasks.
