def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    tp = 0.0 # +
    fp = 0.0 # +
    tn = 0.0 # +
    fn = 0.0 # +

    # print("prediction", prediction)
    # print("ground_truth", ground_truth)
    for i in range(prediction.shape[0]):
        if ground_truth[i] == True and prediction[i] == ground_truth[i]:
            tp += 1
        if ground_truth[i] == True and prediction[i] != ground_truth[i]:
            fp += 1
        if ground_truth[i] == False and prediction[i] == ground_truth[i]:
            tn += 1
        if ground_truth[i] == False and prediction[i] != ground_truth[i]:
            fn += 1

    # print (tp, fp, tn, fn)
    precision = tp / (tp + fp)
    if tp + fn != 0 : recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    if precision + recall != 0: f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    tp = 0.0 # +
    fp = 0.0 # +
    tn = 0.0 # +
    fn = 0.0 # +

    # print("prediction", prediction)
    # print("ground_truth", ground_truth)
    for i in range(prediction.shape[0]):
        if prediction[i] == ground_truth[i]:
            tp += 1
        if prediction[i] != ground_truth[i]:
            fp += 1
        if prediction[i] != ground_truth[i]:
            tn += 1
        if prediction[i] == ground_truth[i]:
            fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return accuracy
