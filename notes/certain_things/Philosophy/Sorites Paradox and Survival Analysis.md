There seems to be a plausible relationship between (a) the idea of a latent evolving hazard in #survival-analysis and (b) the accumulation effect that drives our intuitions from observations of distinct sand grains to observations of a heap.  The classic philosophical puzzle put forward by Sorites.

I've worked on survival analysis in the context of statistics and data science , but I'm putting together this note to arrange my thoughts on the value the perspective has on the classic philosophical puzzle. 

The first thing to observe is how survival analysis is in general a model of the probabilities of state-transition. Moving between alive-dead, sick-well, subscribed-churned, hired-fired. The Framework is quite abstract and therefore widely applicable to the analysis of all state transitions with both clear, distinct and permeable borders between states. 

Traditionally you might see frequentist elaborations of the mechanics of #survival-analysis , but it becomes more interesting from a philosophical stand point when you phrase the model in a #Bayesian fashion. 

In the Bayesian setting the uncertainty expressed in the model regarding measures of risk of state-change can be seen to contribute to the semantic ambiguity relevant in the Sorites setting.  I will try to bring out this connection in the following. 

## The Sorites Paradox
The typical presentation of the sorites issue in philosophy has been as a series of conditional predications as follows: 

- Fa_0
- Fa_0 -> Fa_1
- .
- .
- .
- Fa_{i-1} -> Fa_i
- Therefore: Fa_i

Where we allow that there is an indifference relation for the predication of F over all elements of  the sequence preceding the last entry. Then, it is argued, the last predication fails for some entry (i) in the sequence. The predicate F is said to be suffer from #vagueness .This is purportedly a paradox due to the requirement of logical validity over conditional reasoning caused by the vagueness of F over the course of the sequence. 

Other rephrases of the logical steps truncate the sequence of conditionals by appealing to a *Principle of mathematical Induction* to arrive at the same conclusion. For out purposes it's not crucial which phrasing is applied. The point is just that the the **vagueness** of the predicate F is said to threaten some central tenet of logical validity. 

### Approaches to the Paradox

There have been advocates for acceptance of the Paradox - allowing that there is just a breakdown of logic in the case of these vague predicates. So much the worse for logic. This is quite radical as the prevalence of vague predicates in natural language commits us implicitly to the view that we cannot make distinctions. 

The more plausible approaches to the Paradox seek to establish a reason for rejecting the validity of the conclusion by denying the premises in some way e.g. claiming that the indifference over the nth predication of F and the n-1 case fails. There is a precise point which is a genuine cut-off between F and not F.

The suggestion above is supported somewhat empirically by the reality of borderline cases of predication among reasonable speakers of the same language. People evince hedging behaviour and deliberate vagueness in cases where they avoid commitment to a sharp distinction. "She is sorta cool, nice-ish!". This is the data the philosophical theory needs to explain. 

The theoretical question then becomes - what constitutes **borderline** vagueness? I think this is where we can use survival analysis to elaborate and explain cases of borderline vagueness and empirical cases of disagreement in predication. In particular Bayesian approaches to survival analysis which allow that there is a cut-off point in the sequence, but there is genuine uncertainty where we locate the cut-off point. 

### Perspectives on Probability 

Survival analysis can seem intimidating because it asks us to understand time-to-event distributions from four distinct perspectives. The first familiar density function, the next cumulative density function and its inverse survival function. Additionally we can view the the cumulative hazard function as a transformation of the survival function, and the instantaneous hazard as a discretisation of over intervals of the temporal sequence ranged over by the cumulative hazard. 

It's important to see and understand that these quantities, while appearing to be abstract mathematical objects, can be derived from simple tables which record the subjects in the **risk set** which experience the event of interest at each interval of time. In other words the set of conditional probabilities instance-by-instance over the range of the temporal sequence. This is how we derive the instantaneous hazard quantity.

Different families of probability distribution allow us to encode different structures in the hazards. For instance if we want hazards to peak early in the sequence and decline later in the sequence non-montonically we can use the loglogistic distribution.  If we want to ensure monotonic hazard sequences we can use Weibull distributions. 

### Distinguishing Risk and Uncertainty 

We want to explain the semantic ambiguity of Sorites phenomena by the probabilistic nature of state transitions over additive sequences. However, we won't trade on the uncertainty between distinct models i.e. it's not merely that your model of the sand-to-heap transition is characterised by one probability distributions and mine by another (although it could be). We are interested in divergences of opinion and semantic uncertainty that arises due to the stochastic nature of the phenomena where we share the same model of the phenomena. This reflects a difference in the view of the **risk** not the *model uncertainty*. 

### Cox Proportional Hazards Model 

To make this a bit more concrete consider the cox proportional hazard model. This is a #regression model which aims to characterise the probability of state transition using a statistical model with two components. The baseline hazard: 

$$ \lambda_0 (t) $$

which is combined multiplicatively with an exponentiated weighted linear sum as follows 
$$\lambda_0 (t) \cdot e^{\beta_0 X_0 + \beta_1 X_1 ... \beta_n X_n}$$
In this model  the baseline hazard is a function of the time intervals and we estimate a hazard term for each interval when we fit the model. There is a "free" parameter for the instantaneous hazard at each timepoint. This sequence is the baseline hazard. This latent baseline hazard is akin to an intercept term(s) in more traditional regression models. Individual predictions of the evolving hazard are then determined by how the individuen's covariate profile modifies the baseline hazard. Estimation procedures for this model find values for the baseline hazard and for the coefficient weights $\beta_i$ in our equation. 

With these structures in mind you might be tempted to locate the source of disagreement between people's judgments as stemming from differences in their covariates $X_i$ , or put another way... we see the probability of transition as a function of the same variables, but disagree on the values of those inputs to the function. The benefit of this perspective is that instead of seeing the Sorites Paradox as an error of logical reasoning that needs to be fixed by one or more adjustments to classical logic, we can instead view the phenomena as reflecting disagreement  among latent probabilistic models. 


















