# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:15:08 2024

@author: usouu
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import utils
import covmap_construct as cc

# Initialize a DataFrame to store results
results = []

# Loop through subjects and exercises
for subject in range(1, 2):
    for exercise in range(1, 4):
        identifier = f"sub{subject}ex{exercise}"
        print(f"Processing: {identifier}")

        # Get cm data
        cmdata_pcc = utils.get_cmdata('PCC', identifier)
        cmdata_pcc_alpha = cmdata_pcc['cmpcc_alpha']
        cmdata_pcc_beta = cmdata_pcc['cmpcc_beta']
        cmdata_pcc_gamma = cmdata_pcc['cmpcc_gamma']
        
        # Reshape and combine data
        cmdata_pcc_alpha = cmdata_pcc_alpha.reshape(-1, cmdata_pcc_alpha.shape[1]**2)
        cmdata_pcc_beta = cmdata_pcc_beta.reshape(-1, cmdata_pcc_beta.shape[1]**2)
        cmdata_pcc_gamma = cmdata_pcc_gamma.reshape(-1, cmdata_pcc_gamma.shape[1]**2)
        cmdata_pcc_joint = np.hstack((cmdata_pcc_alpha, cmdata_pcc_beta, cmdata_pcc_gamma))

        # Get labels
        labels = utils.get_label()

        # Split data into training and testing sets
        split_index = int(0.7 * len(cmdata_pcc_joint))
        data_train, data_test = cmdata_pcc_joint[:split_index], cmdata_pcc_joint[split_index:]
        labels_train, labels_test = labels[:split_index], labels[split_index:]

        # Train SVM classifier
        svm_classifier = SVC(kernel='rbf', C=1, gamma='scale', decision_function_shape='ovr')
        svm_classifier.fit(data_train, labels_train)

        # Predict and evaluate
        labels_pred = svm_classifier.predict(data_test)
        accuracy = accuracy_score(labels_test, labels_pred)
        report = classification_report(labels_test, labels_pred, output_dict=True)

        # Store results
        results.append({
            "Identifier": identifier,
            "Accuracy": accuracy,
            **{f"Class_{key}": value['f1-score'] for key, value in report.items() if key.isdigit()}
        })

# Save results to Excel
results_df = pd.DataFrame(results)
results_df.to_excel("results.xlsx", index=False)

print("Processing complete. Results saved to 'results.xlsx'.")
