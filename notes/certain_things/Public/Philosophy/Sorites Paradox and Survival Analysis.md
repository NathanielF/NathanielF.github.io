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

The more plausible approaches to the Paradox seek to establish a reason for rejecting the validity of the conclusion by denying the premises in some way e.g. claiming that the indifference over the nth predication of F and the n-1 case fails. There is a precise point which is a genuine cut-off between F and not F. This position is called Epistemicism - which locates the causes of vagueness in predication in our degrees of ignorance. 

The suggestion above is supported somewhat empirically by the reality of borderline cases of predication among reasonable speakers of the same language. People evince hedging behaviour and deliberate vagueness in cases where they avoid commitment to a sharp distinction. "She is sorta cool, nice-ish!". This is the data the philosophical theory needs to explain. Paradigmatic cases of attributed state-change coupled with paradigmatic cases of hedging. 

The theoretical question then becomes - what constitutes **borderline** vagueness? I think this is where we can use survival analysis to elaborate and explain cases of borderline vagueness and empirical cases of disagreement in predication. In particular Bayesian approaches to survival analysis which allow that there is a cut-off point in the sequence, but there is genuine uncertainty where we locate the cut-off point. Seeing this as a problem of Bayesian modelling allows us to locate the sources of hedging in the components and precision of our model terms through which the propagates the uncertainty in our attribution patterns. 

### Perspectives on Probability 

Survival analysis can seem intimidating because it asks us to understand time-to-event distributions from four distinct perspectives. The first familiar density function, the next cumulative density function and its inverse survival function. Additionally we can view the the cumulative hazard function as a transformation of the survival function, and the instantaneous hazard as a discretisation of over intervals of the temporal sequence ranged over by the cumulative hazard. 

It's important to see and understand that these quantities, while appearing to be abstract mathematical objects, can be derived from simple tables which record the subjects in the **risk set** which experience the event of interest at each interval of time. In other words the set of conditional probabilities instance-by-instance over the range of the temporal sequence. This is how we derive the instantaneous hazard quantity.

Different families of probability distribution allow us to encode different structures in the hazards. For instance if we want hazards to peak early in the sequence and decline later in the sequence non-monotonically we can use the loglogistic distribution.  If we want to ensure monotonic hazard sequences we can use Weibull distributions. 

### Distinguishing Risk and Uncertainty 

We want to explain the semantic ambiguity of Sorites phenomena by the probabilistic nature of state transitions over additive sequences. However, we won't trade on the uncertainty between distinct models i.e. it's not merely that your model of the sand-to-heap transition is characterised by one probability distributions and mine by another (although it could be). We are interested in divergences of opinion and semantic uncertainty that arises due to the stochastic nature of the phenomena where we share the same model of the phenomena. This reflects a difference in the view of the **risk** not the *model uncertainty*. 

### Cox Proportional Hazards Model 

To make this a bit more concrete consider the cox proportional hazard model. This is a #regression model which aims to characterise the probability of state transition using a statistical model with two components. The baseline hazard: 

$$ \lambda_0 (t) $$

which is combined multiplicatively with an exponentiated weighted linear sum as follows 
$$\lambda_0 (t) \cdot e^{\beta_0 X_0 + \beta_1 X_1 ... \beta_n X_n}$$
In this model  the baseline hazard is a function of the time intervals and we estimate a hazard term for each interval when we fit the model. There is a "free" parameter for the instantaneous hazard at each timepoint. This sequence is the baseline hazard. This latent baseline hazard is akin to an intercept term(s) in more traditional regression models. Individual predictions of the evolving hazard are then determined by how the individuen's covariate profile modifies the baseline hazard. Estimation procedures for this model find values for the baseline hazard and for the coefficient weights $\beta_i$ in our equation. 

With these structures in mind you might be tempted to locate the source of disagreement between people's judgments as stemming from differences in their covariates $X_i$ , or put another way... we see the probability of transition as a function of the same variables, but disagree on the values of those inputs to the function. The benefit of this perspective is that instead of seeing the Sorites Paradox as an error of logical reasoning that needs to be fixed by one or more adjustments to classical logic, we can instead view the phenomena as reflecting disagreement  among latent probabilistic models. 

### Complexity of the Heap

One additional perspective on the problem is gained by noting how the Cox proportional model is a __prediction__ model and comes with criteria of model adequacy. How many predictor variables are required to anticipate state transition?  How much variance is explained? If we can gauge the complexity of the prediction task, can the complexity itself explain disagreement? 

### Hierarchical or Meta Vagueness

We've seen now a few different sources of divergences. At the highest level we can appeal to the Knightian distinction between risk and uncertainty, then secondarily to differences in the data used to calibrate risk or thirdly in differences due to estimation strategies and finally in pure prediction complexity.

If divergences are due to complete uncertainty of the appropriate model, then we concede allot to the sceptic and the quantification of any plausible cut point is hopeless. If differences result from the other candidate sources there remains hope for arriving at intersubjective consensus. 

This can be seen in some sense in Bayesian model development workflow with hierarchical survival models.  Instead of imagining agents reasoning if-then style through a sequence of additional sand grains. Let's picture the reasoner working with a latent survival model, negotiating a contract between reality and their linguistic usage.

Hierarchical models in the Bayesian setting are typical and interesting in their own right as they allow for the expression of heterogeneity across individuals. Broadly they involve adding one or more parameters that modify the baseline model equation. We saw earlier that the Cox Proportional hazard model is expressed as
$$\lambda_0 (t) \cdot e^{\beta_0 X_0 + \beta_1 X_1 ... \beta_n X_n}$$

This can be modified as follows:

$$z_{i} \cdot \lambda_0 (t) \cdot e^{\beta_0 X_0 + \beta_1 X_1 ... \beta_n X_n}$$
Where we allow an individual "frailty" term $z_{i}$ is added to the model as a multiplicative factor for each individual in the data set. The terms are drawn from a distribution, often centred on 1, so that the average individual modifies the baseline model not at all... but modifications are expressed as a reduction or increase to the multiplicative speed of state transition. The baseline model can therefore be considered 

Recall that the Bayesian modelling exercise quantifies the probability distribution of all the parameters in the model. A well specified baseline model will mean that less explanatory work needs to be done by the individual frailty terms. A poorly specified model will locate allot of weight in these terms. This is a mechanism which helps quantify the degree of irreducible uncertainty in our attribution patterns derived from our understanding of the paradigmatic cases (our sample). 

### The Bayesian Reasoner

An individual reasoner working with their set of paradigmatic data points y ~ f(X | $\theta$) may fit their model to this data. The variance in the distribution of frailty terms $z_{i} \in \theta$ estimated represents their view of the remaining uncertainty in cases after controlling for the impact of the covariate profiles across the cases. 

These quantities represents disagreements regarding the speed up or slow down in the survival curves...but the survival curves quantifies the probability of transition at each point of accumulation. So a survival model allows us to say precisely at each point of accumulation what is the probability of transition. For any given point in series of accumulating instances, the diversity of individual frailty terms needed to account for the predications determine the quantifiable range of uncertainty in the survival probabilities we derive from the paradigmatic cases. 

Epistemicism about existence of a cut point for vague predicates will always assume the existence of threshold. The picture of Sorites Paradox for the Bayesian reasoner sees them go from uncertainty to uncertainty updating the latent model as they go. Maybe the model converges tightly in some cases, maybe not. Incorporating more paradigmatic instances, more covariate indicators as they develop conceptual clarity on what drives the attribution of state-hood under the evolving or growing pressure to change.  Finding the threshold would not and could not be a solution to the paradox. Any threshold will be context specific and also learned (with uncertainty) relative to the tolerances of the domain. Understanding that paradox as yet another instance of learning in a multifaceted world at least lets us see the problem without requiring torturous modifications to classical logic.
















