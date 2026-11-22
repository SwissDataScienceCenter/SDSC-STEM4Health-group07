# Speaker notes: Decoding the Bioreactor (Team 7), final version

## Slide 1. Decoding the Bioreactor

Hello. In the next few minutes we will show how we predicted thirteen days of a cell culture from very little: the first measurement and the operators’ logbook. The curve on this slide is real data: it is the typical number of living cells over the thirteen days.

## Slide 2. The task

A bioreactor is a tank where living cells are grown; one run of the tank is a batch. We have 61 batches. For 49 we know everything. For 12 we only get the day-0 measurements, the log of what the operators fed the tank, and the tank volume. We have to predict four things for each of the next thirteen days: living cells, total cells, sugar, and the waste product. That is 624 numbers. The score is the average percentage error, so if the truth is 20 and we say 22, that counts as 10 percent.

## Slide 3. One batch, thirteen days

This is what a typical batch looks like, drawn from the 49 batches we know. On top, living cells: they double roughly every day for the first five days, peak at about 30 around day seven, then slowly decline. Below, sugar and waste. Sugar falls as the cells eat it and is topped up by the operators. Waste builds up while the cells grow, then falls because later on the cells eat it back. The shaded bands show how much batches differ from each other. Predicting where inside that band a hidden batch sits is the whole game.

## Slide 4. Even a perfect prediction can’t score zero

The answers we are scored against are themselves measurements, and measuring cells is imprecise. Count the same sample twice and the two counts differ by six to seven percent, at random. Think of a bathroom scale that reads a little high or a little low each time. Even if you knew your exact weight, you couldn’t predict what the scale will show better than its own wobble. Same here: even if we knew the true number of cells exactly, the recorded number would still be about five percent away from it on average. So no model can score much below 5.2 percent on cells. The simplest guess scores 8.7. That means the whole contest on cell counts happens between 8.7 and 5.2. That is why we looked for more information instead of building a fancier model.

## Slide 5. The strategy

This is the order we worked in. First, before building anything, we asked how good a prediction could possibly be. The answer: not much better than about six percent, and we only have 49 batches. So the plan was to find more information, not build bigger models. Second, we looked for that information and found it in the operators’ log. Third, we tried two kinds of model, a statistical one and a simulator of the biology. Both got stuck between 7.4 and 8, and rewriting the biology 21 different ways changed almost nothing. Fourth, we averaged three models that make different mistakes, and then built a simulator that walks sugar through the run day by day. Fifth, we checked: our first model scored 7.03 on held-back batches and 7.02 on the hidden ones. Later, statistical add-ons that looked better on our tests did worse on the hidden batches, so we dropped them.

## Slide 6. The feed log is a sensor.

This is the one idea to remember. The measurements are hidden, but the operators saw them, and they acted on them. They added more feed to tanks with more cells. They topped up sugar only when sugar was low. So the logbook of what they did is an indirect reading of what they saw. We treated every line of the feed log as a sensor.

## Slide 7. Four clues in the operators’ log

Four clues. First, the starting culture: the day-0 cell count is a noisy measurement, but the day-0 waste level and the tank volume also depend on how much starter culture went in, so combining them gives a better estimate of where the batch starts. Second, Feed B and Feed C follow the cell count, so they tell us how many cells there are later in the run. Third, between days two and five the tank gains liquid that nobody logged. We think this is base, added automatically to control acidity, and acidity comes from the waste product. Fourth, the sugar rule. A top-up on days seven to nine happens only if sugar was below 1.95, and on days eleven and twelve the size of the top-up tells us whether sugar was below 0.97. We found these thresholds in the 49 known batches.

## Slide 8. Three models

We ended up with three main models, and the next slides take them one at a time. The first is a simulator of the process, plus a statistical model that learns what the equations get wrong and corrects it. The second uses no biology at all: it is a regression that learns directly how the clues relate to the outcomes. The third is the simulator again, but run a thousand times per batch with slightly different settings. We then keep the runs that agree with the sugar top-up rule and trust those more. On their own each scores about 7.4. But they make different mistakes, so averaging them cancels part of the error and gets us to 7.03. For sugar we later added a fourth model, a day-by-day simulator, which comes after these three.

## Slide 9. Inside the simulator

Our first model is a simulator. It is a small set of equations that plays a batch forward in time, like a weather forecast. Living cells eat sugar and multiply. Growth slows as the tank gets crowded, which we measure as the cell-days accumulated so far. While the culture is young, cells turn part of the sugar into waste. Later, when sugar runs low, they eat the waste back. Some cells die. And every feed adds sugar and dilutes everything in the tank. With one set of settings for all batches this scores 9.9, barely better than the typical batch. When the clues adjust the settings for each batch, for example this one grows three percent faster, it drops to 8.0.

## Slide 10. The simulator in equations

For those who want the equations. On the right is the same thing in plain words. Cells grow when there is sugar, and grow more slowly as the tank gets crowded; we measure crowding as the total number of cell-days so far. Young cells turn part of their sugar into waste. Later, when sugar runs low, they switch to eating the waste. Some cells die. Every feed dilutes the tank and adds sugar. We fit 24 settings, such as the maximum growth rate, to the 49 known batches. Then the clues adjust a few of those settings for each batch. The fitted growth rate means cells double about every 0.93 days, which matches the data.

## Slide 11. The statistical model

The statistical model is a ridge regression, which is a linear regression that is held back from overfitting. It predicts all 52 numbers for a batch at once, as the typical batch shifted up or down by the clues. The clues come in 20 groups, such as the day-0 values, Feed B dosing, or the extra volume. The only things we tune are one weight per group, which says how much to trust it. With 49 batches we can’t afford to tune much more than that. If we point this same model at the simulator’s mistakes, it becomes the correction step of our first model.

## Slide 12. The simulator with uncertainty

The third model runs the simulator many times. First, for every known batch, we find its own small adjustments: grows a bit faster, makes a bit less waste, and so on. Second, we learn how those adjustments relate to the clues. Third, for a new batch, we draw a thousand likely versions and simulate each. Fourth, the key step: we keep only the versions whose sugar would have made the operators do what they actually did. If the log shows a top-up on day 8, a version where sugar stayed high on day 8 is thrown out, along with its cell and waste curves. Fifth, we take the middle of what remains. The numbers at the bottom show both the clues and the log check matter.

## Slide 13. A day-by-day sugar simulator

For sugar we built a fourth model, and it turned out to be our biggest single gain. Instead of predicting the whole curve in one go, it walks through the run a day at a time. It starts from the sugar measured on day zero. Each step predicts tomorrow from today, the feed added per litre and the time between samples; on the known batches one step is right to within three to five percent. Then it uses the operators’ log: on days seven to nine and eleven to twelve, whether they topped up tells us if sugar was above or below their threshold, so any simulated path that breaks that rule is dropped. We run a thousand paths per batch and report the middle one. On the twelve hidden batches it brought the sugar error from 7.23 down to 6.33.

## Slide 14. From models to one prediction

The full recipe in five steps. Read the clues. Run the models: three of them predict all four curves, and the day-by-day simulator predicts sugar. Average them; we average on a percentage scale so that no model dominates, and for sugar most of the weight goes to the simulator. Smooth the cell curves: a batch that is five percent above typical stays about five percent above, so day-to-day wiggles in a prediction are noise. And aim slightly low: with percentage error, guessing too high on a small value costs more than guessing too low. Finally, for sugar, when the top-up rule tells us the value was below a threshold, we keep the prediction below it.

## Slide 15. Error, step by step

This is how the error came down. Predicting the typical batch for everyone: 10.1. The simulator without clues: 9.9, so the equations alone add almost nothing. The simulator with the clues: 8.0. That jump is the clues, not the equations. The pattern-matcher: 7.5. The simulator with a statistical correction: 7.4. The average of our three models: 7.03. And replacing most of the sugar prediction with the day-by-day sugar simulator: 6.86. All of these numbers are measured on the 49 batches where we know the answers, each batch predicted by models that never saw it.

## Slide 16. What each step was worth

This table backs up the strategy. Each row is one step, with the error just before and just after. The big wins all come from using information better: letting the clues adjust the simulator is worth a full point, the statistical correction 0.6, checking against the feed log half a point, and averaging three models 0.3. Smoothing the cells and correcting the third model together are worth 0.06, which is too small to tell apart from chance. The last row is the sugar simulator: 0.17 on the average of four targets, which is about 0.7 on sugar alone. That one also held up on the hidden batches.

## Slide 17. Ablations

This is the technical version of the previous slide, for anyone who wants the exact components. Row one: using the MAPE-optimal point of a log-normal predictive distribution, exp of mu minus sigma squared, and truncating the substrate distribution at the feed-rule thresholds. Row two: covariates on the ODE’s kinetic parameters, a log-linear model in seven process signals with a ridge penalty; this is the biggest single gain. Row three: the hybrid, a kernel ridge with marginal-likelihood weights fitted on the ODE’s log residuals. Row four: replacing four daily volume residuals with one windowed value. Row five: weighting the particles by the likelihood of the logged feed decisions, which is importance sampling. Row six: an equal-weight geometric mean of one member from each family. Row seven is within noise. The last row is the substrate simulator: a one-step ridge model chained from day zero, plus a state-space version fitted by EM with a Kalman smoother; the feed rules enter as truncated-normal constraints with importance weights carried along each path, and the point is a weighted median, which is MAPE-optimal. It is blended on logs with weight 0.7 against the three-family average. All numbers are out-of-fold; the variant names match the experiment log.

## Slide 18. Error by target

By target, for our final submission. The first two number columns are measured on the 49 known batches; the last one is the official score on the twelve hidden batches. The biggest gain is on sugar, from 11.5 for the typical-batch guess down to about 6.3, and that comes from the top-up rule and the day-by-day sugar simulator. Cell counts are near 6.6, which is close to the floor of about 5.2 that the cell counter allows. Waste is the hardest. What remains is concentrated where the values are small: sugar on day eleven, when it is nearly used up, and waste at the end of the run. A small absolute miss on a small number is a large percentage.

## Slide 19. The final submission

This is what we actually submitted at the end. Our first submission, the three-model average, scored 7.02 on the hidden batches, almost exactly what our own tests predicted. After that we submitted several complete versions of the model and looked at the error on each of the four measurements separately. For each measurement we kept the version that did best. Live cells come from a statistical model alone, a kernel ridge regression, total cells from the simulators alone, sugar from the day-by-day simulator, and waste from the three-model average raised by two percent. Every one of these versions was built only from the known batches. But we should be clear: because we chose between them using the hidden scores, 6.71 flatters us a little. On the known batches the same recipe scores 6.94, so that is a fairer guess for new batches.

## Slide 20. How we checked it

How do we know our numbers are real? We tested on batches where we know the answers: hide seven of the 49, build every model on the other 42, predict the seven, and repeat until every batch has been hidden once. That gave 7.03 for our first model, and on the twelve hidden batches it scored 7.02, so the test was right. It was also right that the sugar simulator helps. But it was wrong twice about small statistical add-ons for the cell counts: they looked better on the known batches and did worse on the hidden ones. With only twelve hidden batches, luck can move a score by a few tenths. Our final recipe scores 6.94 on the known batches and 6.71 on the hidden ones; the honest expectation for new batches is somewhere in between, closer to 6.9.

## Slide 21. What we learned

Three takeaways. First, the biggest single gain came from treating sugar the way the operators did, a day at a time, with their top-up rule as a hard constraint: almost a full point off the sugar error on the hidden batches. Second, with only 49 batches, small improvements in our own tests are not reliable: twice, statistical add-ons for the cell counts looked better on known batches and did worse on hidden ones. Third, averaging different models helped more than polishing any one. Overall, what limits the score is the information in the data, and the most useful work was reading the operators’ log carefully. Thank you.
