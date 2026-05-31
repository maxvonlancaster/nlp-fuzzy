# Adversarial Generation of Short Prompts for Improving Feedback Quality

This article describes a research direction that transforms prompt design from a static engineering exercise into an adversarial game. The goal is to generate short questions that systematically improve the quality of human or simulated feedback by increasing information content while reducing emotional bias. The resulting framework sits at the intersection of game theory, natural language processing, and reinforcement learning.

The core motivation is simple but powerful. Traditional surveys and feedback forms rely on fixed questions that often yield responses dominated by sentiment, vague impressions, or single-topic commentary. In domains such as product reviews, customer support, and education, low information density is a greater problem than low volume. In response, the work reframes prompt generation as a competitive interaction between a generator that seeks useful data and an evaluator that penalizes subjectivity.

## From Static Templates to Adversarial Strategies

At the outset, the research considers a finite set of short prompt templates as pure strategies in a zero-sum game. The Generator produces a prompt, and the Evaluator scores the resulting text according to two dimensions: information content and bias. Information is measured through aspect coverage, actionable wording, and named entities, while bias is penalized as the absolute sentiment polarity of the response.

This formulation yields an elegant scientific insight. A prompt is not merely a question; it is a strategy in a game whose payoffs are shaped by the stochastic response distribution of the user. Every prompt choice induces a distribution over feedback texts, and the generator’s utility is expressed as an expected value over that distribution. This probabilistic perspective emphasizes empirical expectations rather than single-shot performance, which is critical when dealing with noisy human responses or LLM simulations.

In the notebook, this game-theoretic foundation is expressed with equations that mirror human intuition. The response space is denoted by $R$ and the prompt space by $P$. For a prompt $p \in P$, the environment produces a feedback text $r \sim P(\cdot|p)$. The generator seeks to maximize

$$u_G(p, \lambda) = \alpha E[I | p] - \beta E[B | p],$$

while the evaluator seeks the opposite. This construction naturally leads to a minimax formulation in which the generator searches for prompts that remain effective even when the evaluator chooses the most difficult weight configuration.

## Defining Quality: Information and Bias

A major contribution of the notebook is the refinement of both information and bias metrics. Rather than relying on a single score, the project splits information into several subcomponents. The aspect score counts mentions of predefined product dimensions such as price, delivery, quality, support, usability, and performance. Actionability is measured by the presence of constructive terms such as "should," "improve," and "fix." Named entity recognition is used as a proxy for specific, concrete content.

These components are combined into a single information metric using a Euclidean-style aggregation, which is more robust than a simple sum. This design decision is important because it discourages the generator from over-optimizing a single axis. A prompt that mentions every aspect superficially earns a lower score than one that combines aspects, suggestions, and concrete entities in a balanced way.

Bias is treated as a separate quantity and is intentionally simple. The sentiment intensity of each response is computed with a sentiment analyzer, and the bias metric is the absolute value of that sentiment. This means that highly positive and highly negative responses are both considered risky. The resulting payoff rewards neutral, objective text while still valuing informative content.

## Parameterized Prompt Generation

The notebook moves beyond fixed templates into a parameterized prompt space. Each prompt is represented by a vector of parameters such as tone, focus, specificity, intent, formality, directness, examples, and length. This vector can be translated into natural language using a modular prompt construction process.

For example, a generator may choose a parameter vector that produces a prompt asking for usability feedback in a friendly tone with a request for specific examples. Another vector may produce a direct, problem-oriented question about delivery experience. The ability to map continuous parameter vectors to prompt text is itself an important scientific advance: it makes the generator’s strategy space differentiable in a practical sense, even though the underlying response process remains discrete and stochastic.

The notebook also explores embedding-space parametrization, where prompts and parameter anchors are represented in a shared semantic space using sentence embeddings. This enables a generator to select new prompts by identifying parameter combinations whose semantic vectors are close to desirable anchors, effectively blending categorical prompt traits with continuous similarity search.

## Adversarial Optimization and Evolutionary Search

A key experimental thread in the notebook is the use of adversarial optimization to search for strong prompt parameters. The research implements a population-based search that samples continuous parameter vectors and evaluates them through simulated feedback. Each candidate prompt is assessed on average information score, average bias, and aspect recall, producing a payoff that reflects the adversarial objective.

The evolutionary strategy is not presented as a final production algorithm, but as a compelling proof of concept. By preserving elite parameterizations, performing crossover, and applying mutation, the search rapidly discovers parameter vectors that outperform random baseline prompts. The notebook records qualitative improvements in average information and recall, showing that the adversarial objective guides search toward prompt families that are both informative and stable.

This stage of the work also highlights the research novelty. Conventional prompt engineering typically relies on manual trial-and-error or exhaustive template search. Here, the generator is allowed to navigate a continuous parameter space under adversarial pressure, producing prompts that are optimized for a dual objective rather than a single quality signal.

## Practical Implementation and Findings

The notebook includes a practical implementation using a simulated feedback generator powered by an LLM, an NLP pipeline built with spaCy and NLTK, and a set of handcrafted aspect keywords. It tests prompt templates such as "What did you like about the product?" and "What should we improve in the product?" before moving to parameterized prompts and neural embedding techniques.

One of the findings from these experiments is that moderate prompt length and explicit focus can increase information density without significantly raising bias. Prompts that combine a narrow focus with a clear request for examples tend to generate responses that mention several aspects while remaining measured in sentiment. In contrast, overly generic prompts often produce shallow or emotionally loaded texts, and overly aggressive prompts risk provoking extreme sentiment.

The article also documents a subtle but important insight: the generator must live in the same space as the evaluator. It is not enough to maximize a generic notion of informativeness. The prompt must be tuned to the evaluator’s metric, which is why the adversarial game is essential. When the evaluator is allowed to change its weights, the generator learns to avoid prompts that are sensitive to those changes, producing strategies that are robust rather than narrowly optimized.

## Novelty and Contribution

The novelty of this work lies in its combination of several ideas into a coherent framework. First, it applies adversarial game theory to the task of prompt generation for human feedback, rather than model alignment or robustness alone. Second, it parameterizes prompts in a way that allows continuous search and semantic matching. Third, it introduces an information metric that balances aspect coverage, actionable suggestions, and entity specificity, while separately penalizing sentiment bias.

Taken together, these elements make the proposed framework more than a prompt crafting toolkit. It becomes a method for transforming open-ended feedback elicitation into a controlled, optimizable process. The generator is no longer a fixed list of questions; it is a strategic agent that chooses prompts according to expected response quality.

## Broader Implications

This research has broader consequences for any system that seeks to collect high-quality text from users. Systems built on adversarial prompt generation could adapt their questions to different domains, avoid leading or emotionally charged wording, and actively shape the feedback loop toward structured, machine-friendly outputs. In educational evaluation, product reviews, or customer support, this means the platform can ask shorter questions while extracting richer data.

The framework also suggests a new way to think about human-machine interaction. Rather than treating prompts as a neutral interface, it treats them as levers that the system can adjust to manage attention and cognitive load. When combined with a robust evaluator, this creates a feedback pipeline that learns not only from responses, but from the quality of the responses themselves.

## Conclusion

The research documented in the notebook demonstrates that short prompt generation can be systematically improved through adversarial and parameterized strategies. By defining a prompt generator and an evaluator as players in a zero-sum game, the work moves beyond intuition-driven prompt design into a setting where robustness can be formalized and optimized.

The practical findings suggest that a carefully chosen parameterization of prompt traits, combined with a dual information-bias metric, can yield prompts that are more likely to elicit useful, balanced feedback. The model’s novelty is not only in the formulas or code, but in the shift of perspective: from fixed questions to adaptive strategies, from single-score metrics to adversarial expectations.

Future work would deepen the model with real user studies, explore dynamic personalization of prompt distributions, and integrate cognitive load into the evaluator’s utility. In the meantime, this article captures the main scientific contributions and shows how adversarial prompt generation can become a viable instrument for improving the quality of feedback in diverse applications.