import os
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import models, transforms
from classification import (Classification, cvtColor, letterbox_image,
                            preprocess_input)
from utils.utils import letterbox_image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, ConfusionMatrixDisplay,f1_score, roc_curve, auc, cohen_kappa_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize

from classification import Classification

test_annotation_path = 'cls_test.txt'

metrics_out_path = "metrics_out"

test_images_dir = ""  # Directory containing test images

class Eval_Classification(Classification):
    def detect_image(self, image):
        image = cvtColor(image)
        image_data = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        image_data = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))
        with torch.no_grad():
            photo = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()
            preds = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()

        return preds

if __name__ == "__main__":
    if not os.path.exists(metrics_out_path):
        os.makedirs(metrics_out_path)

    classfication = Eval_Classification()

    y_true = []
    y_pred = []
    with open(test_annotation_path, "r") as f:
        for line in f:
            label, path = line.strip().split(';')
            y_true.append(int(label))
            pred = classfication.detect_image(Image.open(path))
            y_pred.append(pred)

    y_true = np.array(y_true)
    y_score= np.array(y_pred)


    y_pred = np.argmax(np.array(y_pred), axis=1)


    cm = confusion_matrix(y_true, y_pred)
    print('Confusion Matrix:\n', cm)
    # 绘制混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('metrics_out/without_mapper/ConfusionMatrix1.png', dpi=300)
    plt.show()
    # 计算准确率
    acc = accuracy_score(y_true, y_pred)
    print('Accuracy:', acc)

    # 计算召回率
    recall = recall_score(y_true, y_pred, average='macro')
    print('Recall:', recall)

    # 计算精确率
    precision = precision_score(y_true, y_pred, average='macro')
    print('Precision:', precision)

    # 计算F1值
    f1 = f1_score(y_true, y_pred, average='macro')
    print('F1 Score:', f1)

    # 计算specificity
    FP = cm.sum(axis=0) - np.diag(cm)
    TN = cm.sum() - (FP + np.diag(cm) + cm.sum(axis=1) - np.diag(cm))
    specificity = TN / (TN + FP)
    spec=(specificity[0]+specificity[1]+specificity[2])/3.0
    print('Specificity:', spec)

    # 计算 Kappa 系数
    kappa = cohen_kappa_score(y_true, y_pred)
    print('Kappa Score:', kappa)

    # 计算 MCC 值
    mcc = matthews_corrcoef(y_true, y_pred)
    print('MCC Score:', mcc)

    # 计算 Quadratic Kappa Score
    quadratic_kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    print('Quadratic Kappa Score:', quadratic_kappa)

    with open('metrics_out/without_mapper/metrics1.txt', 'w') as f:
        f.write('Accuracy: {}\n'.format(acc))
        f.write('Recall: {}\n'.format(recall))
        f.write('Precision: {}\n'.format(precision))
        f.write('F1 Score: {}\n'.format(f1))
        f.write('Specificity: {}\n'.format(spec))
        f.write('Kappa Score: {}\n'.format(kappa))
        f.write('MCC Score: {}\n'.format(mcc))
        f.write('Quadratic Kappa Score: {}\n'.format(quadratic_kappa))

    # 绘制ROC曲线
    n_classes = len(set(y_true))
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    fpr_macro = dict()
    tpr_macro = dict()

    for i in range(n_classes):
        fpr_macro[i], tpr_macro[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])

    all_fpr_macro = np.unique(np.concatenate([fpr_macro[i] for i in range(n_classes)]))

    mean_tpr_macro = np.zeros_like(all_fpr_macro)
    for i in range(n_classes):
        mean_tpr_macro += np.interp(all_fpr_macro, fpr_macro[i], tpr_macro[i])

    mean_tpr_macro /= n_classes

    fpr_macro["macro"] = all_fpr_macro
    tpr_macro["macro"] = mean_tpr_macro
    roc_auc_macro = auc(fpr_macro["macro"], tpr_macro["macro"])

    plt.plot(fpr_micro, tpr_micro, label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc_micro))
    plt.plot(fpr_macro["macro"], tpr_macro["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc_macro))

    for i in range(n_classes):
        plt.plot(fpr_macro[i], tpr_macro[i],
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, auc(fpr_macro[i], tpr_macro[i])))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('metrics_out/without_mapper/roc1.png', dpi=300)  # 保存图片
    plt.show()