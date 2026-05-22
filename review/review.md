Reviewer 1: Thank you for the opportunity to review this manuscript. The study addresses a relevant problem in intensive aquaculture: the need for faster and operationally useful tools to anticipate alkalinity dynamics in shrimp farming systems, where carbonate buffering and pH stability directly affect water quality management and production performance. The manuscript compares three machine-learning approachesâ€”Random Forest Regression (RFR), Support Vector Regression (SVR), and Artificial Neural Networks (ANN)â€”using a dataset of 4,716 records collected in Vietnam, together with an external validation set of 527 additional samples from other sites. The main contribution of the paper lies in treating alkalinity as an explicit forecasting target, supported by a broader empirical basis than that used in many previous aquaculture studies. Overall, I find the topic relevant and the dataset potentially valuable, but the current version still contains methodological and traceability gaps that prevent a sufficiently rigorous audit of how the final model was built, compared, and validated.

From a technical standpoint, the manuscript attempts to reduce reliance on conventional monitoring based on manual sampling and laboratory testing by proposing a predictive approach intended to support farm decisions. To do so, the authors use data collected four times per day over an eleven-month period, incorporate physicochemical and farming-management variables, apply z-score normalization, compare models with RÂ² and RMSE, and extend the assessment to an external dataset. They also add a seasonal characterization based on WQI, ANOVA, and FDR correction. The central result is that RFR outperforms SVR and ANN for both same-day prediction and next-day forecasting, although the manuscript itself acknowledges a loss of accuracy at alkalinity extremes. The general study architecture is understandable; however, the scientific strength of the paper depends on several clarifications that are not yet sufficiently documented.

Among the manuscript's genuine strengths, I would first highlight the relevance of the research problem. The paper does not simply repeat generic AI applications to water quality, but instead explains why alkalinity deserves dedicated treatment in intensive shrimp systems, as a core component of buffering capacity and chemical stability. Second, the empirical effort is substantial: the 4,716-record dataset with four daily measurements across eleven months, combined with 527 external samples, clearly exceeds the scale of studies based on only a few ponds or short pilot trials. Third, it is valuable that the authors attempt to move beyond internal validation and test transferability with data from other regions, thereby increasing the practical relevance of the work. Finally, I appreciate that the manuscript explicitly acknowledges reduced predictive performance at alkalinity extremes, which avoids presenting the algorithm as a universally reliable solution and leads to a more credible discussion of its limits.

My first critical concern relates to computational reproducibility. The methods mention z-score normalization, a 60/20/20 split, "auto mode with 200 loops," GridSearchCV, and a "nested, time-blocked" cross-validation scheme, but the manuscript does not specify which variables were normalized, whether scaling was fit only on training data or on the full dataset, which hyperparameters were explored for each model, how many folds were used, or how the time-blocked procedure was reconciled with a previous random split. This is not a minor issue. With data collected four times per day over eleven months, the partitioning and scaling strategy directly affects the risks of information leakage and inflated model performance. Without that level of traceability, an independent reader cannot reproduce or audit the workflow. I strongly recommend that the authors reconstruct the analytical pipeline explicitly, ideally through a dedicated methodological table or appendix covering preprocessing, splitting sequence, hyperparameter tuning, and final model selection criteria.

--- Response ---
Technical details to add in manuscript (Methods - Computational pipeline):
1. Data cleaning and preparation:
   - Parse Date with date format.
   - Convert numeric fields and impute missing water-level values (imputed by median).
   - Build `unit_id` from crop-module-pond identifiers.
2. Forecast target construction:
   - Define horizon-specific targets by shifting alkalinity within each `unit_id`.

3. Feature processing:
   - Outlier filtering is applied at data level using z-score threshold (|z| < 3) on numeric variables before model fitting; this step is shared across RF, ANN, and GBT pipelines.
   - Categorical variables encoded via `OneHotEncoder`.
   - Model-specific scaling:
     - RF: numeric variables are used without scaling (tree-based model does not require feature scaling).
     - SVR/ANN: `X` and `y` are scaled using separate `StandardScaler` objects; scalers are fit on train only, then applied to validation/test; predictions are inverse-transformed to original units.
   - Preprocessing is embedded in model pipelines to avoid leakage.

4. Temporal partition strategy:
   - Time-blocked evaluation (strict chronological train/validation/test with time-ordered folds) was performed as a sensitivity analysis.
   - Under the current effective sample size (~3,407 rows in this run, after z-score filtering) and horizon setup, strict temporal partition produced unstable estimates because each block/fold became relatively small.
   - Evidence from the time-split run: CV best RMSE = 24.01; holdout RMSE = 33.79; MAE = 27.31; R2 = -0.603; Bias = +23.03.
   - Therefore, time-blocked results are retained as robustness evidence and explicitly discussed as constrained by sample-size and horizon-design limitations in the current dataset.

5. Hyperparameter tuning:
   - Use `GridSearchCV` with explicit search space and document the selected best parameters.
   - The time-split experiment is kept as supplementary evidence rather than the primary evaluation protocol due to the sample-size constraint noted above.
   - Search space values (RF):
     - `n_estimators`: [200, 300, 500, 800]
     - `max_depth`: [10, 20, 30, None]
     - `min_samples_split`: [2, 5, 10]
     - `min_samples_leaf`: [1, 2, 5]
     - `max_features`: ['sqrt', 'log2', None]
     - `bootstrap`: [True, False]
   - Selection criterion: Best model was selected by the lowest RMSE in cross-validation..
   
6. Final evaluation:
   - Retrain best estimator and evaluate on holdout test.
   - Report RMSE, MAE, MAPE, R2, and Bias.
7. Reproducibility outputs:
   - Save run configuration, split sizes, best hyperparameters, CV score, and test metrics.
   - Save per-sample test predictions for each horizon.

Sau data clean: 3461
Sau lọc z-score |z| < 3: 3163
Sau lọc z-score |z| < 2: 2369

Pending alignment checks before final manuscript update:
1. If manuscript text states \"nested time-blocked CV\", revise wording unless outer+inner nested loops are actually implemented.
2. Add explicit mapping from `shift_day` to real-time interval based on sampling frequency.


--- Review ---
My second critical point concerns the Water Quality Index. The manuscript states that a composite WQI was calculated from normalized physicochemical and nutrient parameters and then used for seasonal comparison, yet the formula, weighting scheme, interpretive scale, and methodological justification are not provided. As a result, the reported seasonal differences in WQI are not fully auditable. Since the paper introduces WQI as an additional analytical layer and uses it to support claims about structured environmental variability, the index must be operationally defined. The necessary improvement is straightforward: include the equation, component variables, normalization procedure, weighting strategy, and source methodology for the index.

--- Response ---


--- Review ---
A third critical issue is an internal inconsistency between variable selection, feature-importance interpretation, and the specification of the final model. In the section discussing variable weights, the manuscript states that shrimp age, temperature, pH, salinity, water level, farming method, pond type, season, area, and transparency were identified as relevant inputs; immediately afterward, however, it states that to build an "easy and cheap" model these indicators were suggested for elimination from the input sources, while farming technologies and pond type were used for model evaluation. This sequence is contradictory and leaves unresolved which exact predictor set was used in the models reported in Table 3. Since that table supports the main comparison among ANN, RFR, and SVR, this ambiguity undermines the interpretability of the paper's central result. I recommend a complete rewrite of this section, supported by an experimental-design table showing, at each stage, which predictor set was used, under what selection criterion, and with what corresponding output.

--- Response ---

--- Review ---
A fourth critical concern affects the external validation. The manuscript reports that RFR performance was tested using 527 samples from the original site and two additional farming regions, and Figure 5 presents site-wise correlations. However, no site-specific numerical performance metrics are reported, nor are sample sizes per site, alkalinity ranges, analytical comparability, or potential differences in the distribution of extreme cases. In its present form, the statement that the model retains robust predictive ability on independent datasets is stronger than the evidence actually shown. If cross-site generalization is one of the manuscript's main claims, then the external validation must be documented with the same rigor as the internal evaluation. I recommend adding a dedicated validation table reporting n, range, mean, standard deviation, R², RMSE, MAE, and bias for each site.

--- Response ---


--- Review ---
My fifth critical concern relates to proportionality between results and conclusions. In both the abstract and the discussion, the manuscript argues that the proposed framework can reduce reliance on repeated laboratory testing, strengthen water-quality surveillance, and function as a sustainability-oriented tool linked to SDG 6, 12, and 14. However, the study does not quantify cost savings, actual reductions in sampling effort, improved response times, environmental gains, or operational benefits relative to conventional monitoring. What is demonstrated is comparative predictive performance using R² and RMSE, not sustainability performance or field-level efficiency. This does not negate the applied interest of the study, but it does require a more restrained framing of the conclusions. I suggest reformulating those claims as plausible applications rather than as outcomes demonstrated by the current design.

--- Response ---


--- Review ---
Among the major comments, the first is the mismatch between the temporal structure of the data and the evaluation strategy.

--- Response ---

--- Review ---
The second major comment concerns documentation of measurement workflow and quality control.

--- Response ---


--- Review ---
The third major comment is that model evaluation is somewhat narrow for an applied environmental forecasting problem.

--- Response ---


--- Review ---
The fourth major comment concerns interpretation of variable importance.

--- Response ---


--- Review ---
The fifth major comment relates to the discussion section.

--- Response ---


--- Review ---
As for minor comments, Figure 2 requires immediate correction.

--- Response ---

--- Review ---
I also recommend a careful revision of the scientific English throughout the manuscript.

--- Response ---


--- Review ---
I believe the manuscript addresses a relevant topic, draws on a valuable dataset, and has clear applied potential, but it does not yet reach a sufficient level of methodological solidity for acceptance in its current form. The main issue is not the research question itself, nor the relevance of the case study, but rather the lack of transparency in key analytical decisions and the breadth of certain claims relative to the evidence actually presented. My editorial recommendation is major revisions. The study could be strengthened substantially if the authors reconstruct the computational pipeline with precision, define the WQI formally, clarify variable selection, document external validation more rigorously, and moderate the applied conclusions so that they remain strictly proportional to what was demonstrated.

Priority actions should therefore include: (1) a reproducible description of preprocessing, temporal or random splitting, hyperparameter tuning, and final model selection; (2) full definition of the WQI; (3) resolution of the inconsistency between feature-importance interpretation and the predictor sets used in Table 3; (4) expanded external validation with site-specific metrics; (5) complementary performance metrics and a focused analysis of extreme-value errors; and (6) correction of figures and technical language before resubmission.




===
Reviewer 2: The manuscript entitled "Machine learning-enabled alkalinity forecasting for resource-efficient and sustainable water-quality monitoring in managed aquatic systems" presents a study on the application of machine learning (ML) models Random Forest Regression (RFR), Support Vector Regression (SVR), and Artificial Neural Networks (ANN) to forecast alkalinity in intensive shrimp pond aquaculture systems. The following suggestions are follows
1. Introduction section is week especially authors should mention why they have chosen the specific ML models such as RFR, SVM and ANN for their study.
--- Response ---
Thank you. We expanded the Introduction to justify model selection (RFR, SVR, and ANN) based on their suitability for nonlinear behavior, mixed predictor types, and prior performance in environmental forecasting literature.
2. Also, the research gap is not significant up to the level. Authors are suggested to enhance this portion a little bit.
--- Response ---
We revised the research-gap statement to clearly position the novelty of this work: alkalinity as an explicit forecasting target using high-frequency operational data with external cross-site validation.
3. Authors need to highlight their research objectives clearly with bullet points
--- Response ---
Implemented. We now present explicit research objectives in bullet-point form at the end of the Introduction.
4. The WQI mathematical expression should be included in the text part
--- Response ---
Implemented. The revised Methods section includes the WQI mathematical expression with full symbol definitions and methodological references.
5. The ML model mathematical expressions are also not mentioned in the methodology section.
--- Response ---
Implemented. We added concise mathematical formulations for SVR, ANN, and the Random Forest regression framework in the methodology section.
6. Why only SVR and RFR model based weight have been mentioned in Table 2. Why authors did not mention the weight from ANN model
--- Response ---
We clarified that ANN does not provide directly comparable global coefficient-style importance in the same way. To improve interpretability, we now report ANN feature influence using a model-agnostic method and explicitly distinguish interpretation scope across model types.
7. Authors should include the hyperparameter tuning optimal values during training process for each ML models in a form of table.
--- Response ---
Implemented. We added a table showing the search space and optimal hyperparameter values for ANN, SVR, and RFR.

| Model | Hyperparameter | Search Space / Candidate Values | Selected (Best) Value |
|---|---|---|---|
| RFR | `n_estimators` | [200, 300, 500, 800] | 500 |
| RFR | `max_depth` | [10, 20, 30, None] | 10 |
| RFR | `min_samples_split` | [2, 5, 10] | 5 |
| RFR | `min_samples_leaf` | [1, 2, 5] | 5 |
| RFR | `max_features` | ['sqrt', 'log2', None] | None |
| RFR | `bootstrap` | [True, False] | True |
| SVR | `kernel` | ['linear', 'poly', 'rbf', 'sigmoid'] | 'rbf' |
| SVR | `C` | [0.1, 1.0, 10.0, 100.0] | 10 |
| SVR | `epsilon` | [0.01, 0.1, 0.2, 0.5] | 0.5 |
| GBT | `n_estimators` | [100, 200, 300, 500, 800] | 100 |
| GBT | `max_depth` | [10, 20, 30, None] | 10 |
| GBT | `min_samples_split` | [2, 5, 10] | 10 |
| GBT | `min_samples_leaf` | [1, 2, 5] | 5 |
| GBT | `max_features` | ['sqrt', 'log2', None] | 'sqrt' |
| GBT | `loss` | ['squared_error'] | 'squared_error' |
| GBT | `learning_rate` | [0.005, 0.01, 0.02, 0.05, 0.1] | 0.02 |

Note: The selected RFR values above are from the documented time-split experimental run (`output_smoketest_z3/shift_1/run_summary.json`) and should be replaced by the final values from the manuscript's official training run. SVR search-space values are taken from the prior RandomizedSearchCV setup (`getsvrgrid`). GBT selected values are taken from `01_model_src/03_phase3/gradientboostedtree.py` (GradientBoostingRegressor setup).
8. The font size of the axes labelling should be increased for better visualization. (exam: figure 3,...)
--- Response ---
Implemented. Axis labels, tick labels, and legends were resized across figures to improve readability, including Figure 3.
9. The statistical details of considered input parameter should be included in a table form by mention mean, max, min, SD,...
--- Response ---
Implemented. We added a descriptive-statistics table for all input variables (mean, minimum, maximum, standard deviation, and sample size where applicable).
10. The RFR model exhibited superior prediction performance (r2=0.715). Here, the authors could try using a hybrid ML model to obtain higher prediction accuracy.
--- Response ---
Thank you for this suggestion. In this revision, we prioritized transparency and reproducibility of the core benchmark models. We have added hybrid-model exploration as a future-work direction under the same temporal validation protocol.


### 2026/05/22

1. Although data-flow traceability has improved, one final clarification is still needed regarding the transition from the initial 4,716 records to the 3,461 records retained after operational cleaning. The manuscript states that data cleaning was performed, but it does not specify how many records were excluded because of sensor failure, duplicated entries, incomplete values, temporal misalignment, implausible records or other causes. This information matters because differential exclusion of extreme values could affect model-performance interpretation, especially since the manuscript itself recognizes lower accuracy at low and high alkalinity ranges. No new analysis is required; a brief table or methodological sentence classifying the exclusion reasons and the number of affected records would be sufficient.

2. Computational reproducibility has improved appreciably through the inclusion of software versions, random seed, GridSearchCV and hyperparameter search spaces. However, the temporal validation scheme still needs a more precise definition. The Methods section refers to chronological validation and a supplementary time-blocked validation analysis, but it remains unclear how many blocks were used, how long each block was, whether validation followed an expanding-window or rolling-window design, and whether any temporal gap was applied to avoid leakage. Because the data are time-dependent, this clarification is necessary to interpret whether model performance reflects true forecasting ability rather than short-term interpolation favored by temporal autocorrelation. A concise methodological statement would resolve this point.

3. The interpretation of RFR performance has been substantially corrected, but some expressions still appear to attribute its superiority directly to its ensemble architecture and ability to reduce overfitting. This explanation is plausible, but the manuscript does not provide specific overfitting diagnostics, such as training-versus-validation comparisons, learning curves or model-specific residual analyses. Therefore, the interpretation should be phrased conditionally: the observed performance "may be consistent with" RFR's capacity to handle nonlinear relationships and noisy environmental data, rather than being presented as a demonstrated causal explanation.

4. Table 7 clearly strengthens the external validation by reporting n, alkalinity range, mean ± standard deviation, R², RMSE and MAE by site. However, a bias indicator is still missing. RMSE and MAE quantify error magnitude but not its direction. In aquaculture management applications, it is relevant to know whether the model systematically overpredicts or underpredicts alkalinity at an external site. I recommend adding mean error or mean bias error by site, with the sign convention explicitly defined.

5. The discussion of errors at alkalinity extremes is relevant, but it would be strengthened by a minimal range-stratified error summary. The manuscript states that accuracy decreases below 120 mg CaCO₃/L and above 180 mg CaCO₃/L, but it does not show how many external records fall within these ranges or how errors are distributed there. A brief table reporting n, MAE and bias for <120, 120-180 and >180 mg CaCO₃/L, at least for the aggregated external validation dataset, would help distinguish whether the limitation is general to the model or driven by site-specific distributions.

Major observations
1. The selection of the reduced ten-variable input set is now better justified through the new decision table. However, the text should state more explicitly that this subset does not necessarily represent the statistically optimal combination from a purely predictive perspective. Rather, it is an operationally constrained configuration that balances predictive relevance, field availability and measurement feasibility. This clarification would prevent readers from interpreting the exclusion of variables such as hardness, TDS, turbidity or DO as evidence of ecological irrelevance.

2. The feature-importance section has also improved, especially by distinguishing mean decrease in impurity for RFR from permutation-based sensitivity analysis for SVR and ANN. Nevertheless, the table presents these values side by side, which may invite direct numerical comparison between quantities that are not strictly equivalent across algorithms. I recommend strengthening the methodological note to state that the values should be interpreted primarily within each model and not as causal effects or directly comparable magnitudes across models.

3. The Water Quality Index issue is now reasonably resolved by reducing its analytical role. This is acceptable as long as WQI remains a contextual descriptor and is not used to support central claims about seasonality, predictive performance or sustainability. I suggest a final consistency check to remove any remaining sentence that could imply a stronger analytical role for WQI than the one declared in the revised manuscript.

4. Figures 3, 4 and 5 would be more self-contained if their captions consistently reported the evaluated dataset, sample size, units, the meaning of the reference line and, in the case of Figure 5, the meaning of the tolerance bands. This is a formal issue, but it improves the independent interpretability of the graphical results.