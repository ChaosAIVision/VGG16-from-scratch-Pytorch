from typing import List, Dict, Union,Optional
import numpy as np

def calculate_accuracy(
    y_true: List[int], 
    y_pred: List[int], 
    is_all: bool = False, 
    sample_weight: Union[List[float], None] = None
) -> Union[float, Dict[int, float]]:
    """
    Calculate accuracy from classification predictions.

    Parameters
    ----------
    y_true : List[int]
        Ground truth (correct) labels.

    y_pred : List[int]
        Predicted labels, as returned by a classifier.

    is_all : bool, default=False
        If True, return the fraction of correctly classified samples (overall accuracy).
        If False, return accuracy for each class as a dictionary.

    sample_weight : List[float] or None, default=None
        Sample weights.

    Returns
    -------
    Union[float, Dict[int, float]]
        If is_all == True, return the fraction of correctly classified samples (float).
        Otherwise, return a dictionary where the keys are the class labels and the values are the accuracy for each class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        if len(sample_weight) != len(y_true):
            raise ValueError("Sample weights must be the same length as y_true")
    else:
        sample_weight = np.ones_like(y_true, dtype=float)
    
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be the same")

    if is_all:
        correct = y_true == y_pred
        if sample_weight is not None:
            return float(np.average(correct, weights=sample_weight))
        else:
            return float(np.mean(correct))
    else:
        classes = np.unique(y_true)
        accuracies = {}
        
        for cls in classes:
            true_positives = np.sum((y_true == cls) & (y_pred == cls))            
            total_samples = np.sum(y_true == cls)
            class_accuracy = true_positives / total_samples if total_samples > 0 else 0
            accuracies[cls] = class_accuracy
        
        return accuracies


def confusion_matrix(
    y_true: Union[list, np.ndarray],
    y_pred: Union[list, np.ndarray],
    labels: Optional[Union[list, np.ndarray]] = None,
    sample_weight: Optional[Union[list, np.ndarray]] = None,
    normalize: Optional[str] = None
) -> np.ndarray:
    """
    Compute the confusion matrix to evaluate the accuracy of a classification.

    The confusion matrix is a table used to describe the performance of a classification model.
    It is a matrix where the element at row i and column j represents the number of observations
    actually in class i but predicted as class j.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.

    labels : array-like of shape (n_classes), default=None
        List of labels to index the matrix. If None, all labels that appear in y_true or y_pred are used.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns) conditions or the entire population.

    Returns
    -------
    C : ndarray of shape (n_classes, n_classes)
        Confusion matrix where C[i, j] is the number of samples with true label being i-th class
        and predicted label being j-th class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    else:
        labels = np.asarray(labels)
        if len(labels) == 0:
            raise ValueError("'labels' should contain at least one label.")
        if not np.isin(y_true, labels).all() or not np.isin(y_pred, labels).all():
            raise ValueError("At least one label specified must be in y_true or y_pred")

    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight)
        if len(sample_weight) != len(y_true):
            raise ValueError("Sample weights must be the same length as y_true")

    label_to_index = {label: index for index, label in enumerate(labels)}
    y_true_indices = np.array([label_to_index[label] for label in y_true])
    y_pred_indices = np.array([label_to_index[label] for label in y_pred])

    cm = np.bincount(
        y_true_indices * len(labels) + y_pred_indices,
        weights=sample_weight,
        minlength=len(labels) ** 2
    ).reshape(len(labels), len(labels))

    if normalize == 'true':
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        cm = cm.astype(float) / cm.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        cm = cm.astype(float) / cm.sum()

    return np.nan_to_num(cm)

def calculate_precision_recall(cm: np.ndarray, categories: list = None, average: str = None) -> dict:
    """
    Calculate precision and recall from confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape (n_classes, n_classes).

    labels : list, default=None
        List of class labels. If None, classes will be labeled as 0, 1, ..., n_classes-1.

    average : str, default=None
        If 'all', returns the average precision and recall across all classes.
        If None, returns precision and recall for each class.

    Returns
    -------
    results : dict
        A dictionary with precision and recall for each class or averaged across all classes.
    """
    n_classes = cm.shape[0]
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)

    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0

    if average == 'all':
        avg_precision = precision.mean()
        avg_recall = recall.mean()
        return {
            'average_precision': avg_precision,
            'average_recall': avg_recall
        }
    else:
      
        if len(categories) != n_classes:
            raise ValueError("Number of labels must match the number of classes in the confusion matrix.")

        precision_dict = {categories[i]: precision[i] for i in range(n_classes)}
        recall_dict = {categories[i]: recall[i] for i in range(n_classes)}

        return {
            'precision': precision_dict,
            'recall': recall_dict
        }



