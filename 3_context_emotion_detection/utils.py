# THIS SCRIPT WILL SET UP SOME PUBLIC VARIABLES

# IMPORT PACKAGES
from sklearn.metrics import f1_score, precision_score, recall_score

print("Loading utils module")
label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}

# SET UP THE METRICS
metrics = {
    "f1_e": (lambda y_test, y_pred:
             f1_score(y_test, y_pred, average='micro',
                      labels=[emotion2label['happy'],
                              emotion2label['sad'],
                              emotion2label['angry']
                              ])),
    "precision_e": (lambda y_test, y_pred:
                    precision_score(y_test, y_pred, average='micro',
                                    labels=[emotion2label['happy'],
                                            emotion2label['sad'],
                                            emotion2label['angry']
                                            ])),
    "recoll_e": (lambda y_test, y_pred:
                 recall_score(y_test, y_pred, average='micro',
                              labels=[emotion2label['happy'],
                                      emotion2label['sad'],
                                      emotion2label['angry']
                                      ]))
}
