# credit-risk-classification

# tl;dr
**The goal of data analytics** must include providing as clear a picture of reality as is possible without marginalizing opportunities for the entire community. We do not currently have a credit sniffing minority report to tell us which applicant may lose their job, get a divorce, or become ill, and current indicators lead us to believe we likely do not want to live in that reality.

## Overview of the Analysis

**Explain the purpose of the analysis.** The purpose of the analysis is to predict whether a loan will be healthy or risky based on 7 discrete variables: loan_size, interest_rate,	borrower_income,	debt_to_income,	num_of_accounts,	derogatory_marks, and	total_debt.  

**Explain what financial information the data was on, and what needed to be predicted.** Lending_data.csv contains 77,536 records with 8 fields: loan_size,	interest_rate,	borrower_income,	debt_to_income,	num_of_accounts,	derogatory_marks,	total_debt, and	loan_status. We need to predict the loan_status field.  

**Provide basic information about the variables to be predicted (e.g., `value_counts`).** The only information we are given is that a loan_status of "0" indicates a loan is healthy (75,036 records) while "1" indicates a loan is risky (2,500 records).  

**Describe the stages of the machine learning process encountered as part of this analysis.** We train our model to learn how to predict the loan_status field based on the content of the other 7 fields using the `train-test split` process. We split the records into a training set that studies the pattern which emerges from the data and tests the newly created model on the remaining testing set to find our level of accuracy predicting healthy vs risky loans.  

**Briefly touch on methods used (e.g., `LogisticRegression`, or any resampling method).** Using `LogisticRegression`, we created a model where we fit the 7 characteristics of a record marked "0" or "1", predicted which of the remaining records would be healthy or risky, tested our model, and displayed the results via the `confusion_matrix` and `classification_report`. We then used the `RandomOverSampler` module from the `imbalanced-learn` library to resample the data, artificially adjusting the labels to have an equal number of data points, and ran the `LogisticRegression` module again, refitted the model, made our new prediction, tested our new model, and displayed the new results via the new `confusion_matrix` and `classification report`.  

## Results

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.

                      confusion_matrix
                      [18658,   107]
                      [   37,   582]

                    * True Healthy = 18,658
                    * False Risky = 107
                    * False Healthy = 37
                    * True Risky = 582
                    * Precision: 582 / (582 + 107) = 0.84470247 or 84%
                    * Recall:    582 / (582 + 37) = 0.94022617 or 94%
                    * F1-Score:  2 * (0.84470247 * 0.94022617) / (0.84470247 + 0.94022617) = 0.889908258 or 89%
                    * Accuracy:  18,658 + 582 / 19,384 = 0.99257119 or 99%

                      classification_report
                      precision recall  f1-score   support
              0       1.00      0.99      1.00     18765
              1       0.84      0.94      0.89       619
        accuracy                          0.99     19384
      macro avg       0.92      0.97      0.94     19384
    weighted avg      0.99      0.99      0.99     19384

    Of 19,384 records tested, our first model correctly identified 18,658 of the 18,695 "0", mislabeling 37 and correctly labeling pertneer 100% of the healthy loans, correctly identifying 582 of the 689 "1", mislabeling 107 and correctly labeling 84% of the risky loans (our `Precision Score`[^1]). This result makes sense because the original dataset only has a fraction as many "1" records as "0" records: y.value_counts() = 75036 "0" and 2500 "1". Despite mislabeling a higher percentage of "1" than "0", 582 of the 619 records labeled "1" did belong to that category, giving us a `Recall Score` of 94%[^1].  When class distributions are imbalanced, our `F1 Score` of 89%[^1] best describes the overall accuracy of this model.[^2]
  * Model 1 Viability: **This model can be trusted to predict which loans will be healthy vs risky with 89% accuracy.** 

* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

                      confusion_matrix
                      [55957   314]
                      [  286 55985]

                    * True Healthy = 55,957
                    * False Risky = 314
                    * False Healthy = 286
                    * True Risky = 55,987
                    * Precision: 55,985 / (55,985 + 314) = 0.99442264 or 99%
                    * Recall:    55,985 / (55,985 + 286) = 0.99491745 or 99%
                    * F1-Score:  2 * (0.99 * 0.99)/(0.99 + 0.99) = 0.99 or 99%
                    * Accuracy:  55,957 + 55,985 / 112,542 = 0.99466866 or 99%

                      classification_report
                      precision  recall  f1-score  support
              0       0.99        0.99     0.99     56271
              1       0.99        0.99     0.99     56271
        accuracy                           0.99    112542
      macro avg       0.99        0.99     0.99    112542
    weighted avg      0.99        0.99     0.99    112542
                  
    Of 112542 records tested, our second model "correctly" identified 55,957 of the 56,271 "0", mislabeling 286 and "correctly" labeling pertneer 99% of the healthy loans and "correctly" identified 55,985 of the 56,299 "1", mislabeling 314 and "correctly" labeling 99% of the risky loans (our `Precision Score`[^1]). 55,985 of the 56,271 records labeled "1" did belong to that category, giving us a 99% `Recall Score`[^1].  Now that class distributions are artificially balanced, our 99% `Accuracy Score`[^1] best describes the overall "accuracy" of this model.[^2] The original dataset of 75,536 records was imbalanced with a mere 2,500 records labeled "1" (risky). With only 3% of the records available for the `test-train split` (2,500 / 75,536 = 0.0330968), the fact that the original model could be trusted to predict which loans would be risky with an 89% accuracy rate is a testament to the power of the `LogisticRegression` model. To correct the imbalance, our `RandomOverSampler` bloated the file by 45% ((112,542 - 75,536) / 75,536 = 0.451481) because it "duplicates examples from the minority class in the training dataset and can result in overfitting for some models."[^3] The 37,0006 clones represent over 14 times the number of actual risky loans (112,542 - 75,536 = 37,006 / 2,500 = 14.8024), rendering this model a self-congratulating exercise with no practical use.
  * Model 2 Viability: **This model should NOT be trusted to predict any real world activity.** 

## Summary

Whether unsupervised or supervised, all machine learning risks losing its relevance because we can simply keep running new models until we find the one we like. In our search for a model nearing 100% accuracy, we have so manipulated the sample data in this exercise that our model no longer provides any real world solutions. "Because machine learning offers a high level of modeling freedom, it tends to overfit the data."[^4] Credit modeling is an excellent example of this danger: We do not need a machine to tells us that an applicant with a lot of money, a little debt, and a history of paying their bills who is financing a fixed asset with solid equity is a good credit risk, but we will continue the search for a magic algorithm to bridge the gap for marginal cases and increasingly risky ventures. In our quest to predict the future, we are tempted to conflate causation and correlation: "A machine learning model, unconstrained by some of the assumptions of classic statistical models, can yield much better insights that a human analyst could not infer from the data."[^4] An example of how seemingly unrelated information can cross categories is the now standard but often unknown practice of considering credit scores when setting insurance rates: "The use of credit in car insurance pricing is controversial for a number of reasons, chief among them is that credit has nothing to do with how you drive."[^5] While overfitting data and crossing categories can result in statistical errors and common nuisances, there is a darker side to eradicating risk; A recent paper from the Federal Reserve Board found that minority applicants "are less likely than white applicants to receive algorithmic approval from race-blind government automated underwriting systems."[^6] 

[^1] https://medium.com/@kennymiyasato/classification-report-precision-recall-f1-score-accuracy-16a245a437a5
[^2] https://medium.com/analytics-vidhya/accuracy-vs-f1-score-6258237beca2#:~:text=Accuracy%20can%20be%20used%20when,to%20evaluate%20our%20model%20on
[^3] https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
[^4] https://www.moodysanalytics.com/risk-perspectives-magazine/managing-disruption/spotlight/machine-learning-challenges-lessons-and-opportunities-in-credit-risk-modeling
[^5] https://www.forbes.com/advisor/car-insurance/rate-increase-poor-credit/
[^6] https://www.federalreserve.gov/econres/feds/files/2022067pap.pdf


# Module 20 Challenge for Vanderbilt Data Analytics January 2024

There are 3 files and 1 folder located in this repository, as follows: This README file is in the main directory with credit_risk_classification.ipynb, and lending_data.csv is located in the Resources folder.

**Special Recognition:** Ahmad Mousa is as passionate and caring as he is knowledgeable, and Joshua Steier never hesitates to help walk us through challenges. Ahmad is the epitome of patience and has now explained to me how to read the confusion_matrix on 3 separate occasions. Now that I have used it extensively for this challenge, it is locked into my long term memory, and I am free to use his time more wisely. I always do participate in office hours and study groups while working on projects with collaborators including but not necessarily limited to the following peers: Ilknur Sekmen, Justin Ibeh, Karson Kosek, Kiara Shannon, Luisa Dinwiddie, Margo Berry, Morgan Escue, Morgan Foge, Nathan Johnson, William Brewer, Andrew Clifft, Angela Reed, and Josh Gibson, and I did spend my time in stackoverflow.com, w3schools.com, geeksforgeeks.com, github.com, and bing's new copilot whom I currently consider > google ai; however, the bootcamp has added a new Xpert Learning Assistant, and my life is now complete. Thank you for this resource!

The Google Drive location for this file is as follows: https://drive.google.com/drive/folders/1wCnKcVTe8KK_kTDB7Ftbr18TbKLxdEnN?usp=sharing
