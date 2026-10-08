# Assignment 1 Report

- **Name**: Rubén Segura
- **Student ID**: 19061
- **Email**: rsegura.ieu2022@student.ie.edu
- **Group**: [BBADBA 5B]

## Dataset

I'm using the Online Shoppers Purchasing Intention dataset from the UCI Machine
Learning Repository (dataset 468, donated in 2018 by Sakar et al.). It's a single
1 MB CSV, which I committed to `data/` so training doesn't depend on UCI being up.

Each row is one browsing session on an e-commerce site, and the sessions were
collected so that each one belongs to a different user across a one-year period —
that way no single campaign or heavy user dominates the data. There are 12,330
rows and 18 columns: 17 features plus the binary target, `Revenue`, which says
whether the session ended in a purchase. There are no missing values. Only 1,908
sessions (15.5%) converted, so the classes are fairly imbalanced.

I picked it for three reasons. It's a real binary decision a business would
actually care about, so the framing isn't artificial. The features leave real room
for engineering rather than just dropping them into a model: there are paired
count/duration columns for three page types, Google Analytics metrics like bounce
and exit rates, and a month column that lets me split by time instead of randomly.
And the class imbalance means the decision threshold genuinely matters, which is
exactly what I wanted to build the dashboard around.

## Business / real-life framing

**The scenario.** An online store that scores sessions as they happen. After
about 100 seconds of browsing, the model estimates how likely the session is to
end in a purchase. If that probability falls below my decision threshold, the
site offers a 10% discount coupon on the spot. The numbers I'm assuming: an
average basket of €80, a 35% gross margin (so €28 per conversion), a coupon
worth €8, and a 5% chance that the coupon actually recovers someone who wasn't
going to buy.

**What the errors cost.** The coupon fires on sessions predicted *not* to
convert, which flips the usual intuition about which mistake hurts:

| Case | Action | Impact |
|---|---|---|
| Buys, predicted buy | none | €0 (margin untouched) |
| Buys, predicted no-buy | coupon | −€8 (discount on a sale I already had) |
| No buy, predicted no-buy | coupon | +€1 (5% × €20 net margin) |
| No buy, predicted buy | none | €0 (missed opportunity) |

Net value = (correctly spotted non-buyers × €1) − (buyers I mislabelled × €8).

**First consequence: which features I'm allowed to use.** If the prediction
happens mid-session, I can only use what the site actually knows at that moment.
The page counters and durations update in real time, so those are fine.
BounceRates and ExitRates are historical metrics of the pages visited, not
something derived from how this session ended, so I kept them too. PageValues is
the problem: it's computed from pages visited *before completing a transaction*,
which means it carries information about the outcome. I treat it as a likely
leak, train the models with and without it, and report the difference in the
modelling section.

**Second consequence: the split has to be time-based.** The model would be
trained on past traffic and deployed on future traffic, so I split on the Month
column (February to December), training on the earlier months and testing on the
last ones. A random split would shuffle seasonality and campaigns together —
SpecialDay is tied to specific dates — and would give me a score that looks
better than anything I'd see in production.

**Third consequence: what sets the threshold.** Accuracy is useless here, since
always predicting "no purchase" is right 84.5% of the time. I report AUC and
average precision to compare the three implementations, but I pick the threshold
by maximising the net value in the table above. With errors costing €8 against
€1 gains, the optimum is cautious: only intervene when the evidence of no
purchase is strong. Chasing recall on the "no purchase" class without looking at
the cost would mean handing out margin to people who were already buying.


## Data preparation & feature engineering

*What you engineered and why, and any data-quality decisions you made along
the way — e.g. "segment X had defective data, so I excluded it and used a
population-average default for scope Y at inference time; the impact of
that choice is Z."*

## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
