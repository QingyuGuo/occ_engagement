import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import random
import time
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report
from utils import *
import argparse
import logging
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


LOG_HOME = './'

parser = argparse.ArgumentParser(description='Fine Tune Model for Cognitive Engagement Classification.')
parser.add_argument('--category', type=int, default=3,
                    help="how many categories.")
parser.add_argument('--batch_size', type=int,
                    default=16, help='The batch size.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--seed', type=int, default=0, help='random number seed')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate.')
parser.add_argument('--weight_decay', type=float,
                    default=1e-3, help='penalty on parameters.')
parser.add_argument('--patience', type=int, default=50,
                    help='Epoch number that wait for stop.')
parser.add_argument('--gpu', type=int, default=0, help='cuda device.')
parser.add_argument('--model_name', type=str,
                    default='distilbert-base-uncased', help='Name of the model')
args = parser.parse_args()


enable_cuda = True
feature_extract = True


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if enable_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda:{}'.format(args.gpu))
else:
    device = torch.device('cpu')


logging.basicConfig(level=logging.INFO, filename=LOG_HOME+'{}-seed-{}-lr{}-wd{}.log'.format(
    args.model_name, args.seed, args.lr, args.weight_decay),
    format='%(asctime)s - %(process)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(args)


def train_eopch(loader):

    train_loss = []

    for (X, Y) in tqdm(loader):

        model.train()
        input_ids = X[0].to(device=device)
        attention_mask = X[1].to(device=device)
        Y = Y.to(device=device)
        optimizer.zero_grad()

        y_pred = model(input_ids=input_ids,
                       attention_mask=attention_mask).logits

        loss = loss_criterion(
            y_pred.view(-1, args.category), Y.view(-1).long())

        loss.backward()

        optimizer.step()

        train_loss.append(loss.detach().cpu().numpy())

    mean_loss = sum(train_loss)/len(train_loss)

    return mean_loss


def test_epoch(loader):

    val_loss = []

    y_preds = []
    y_labels = []
    y_probs = []

    for (X, Y) in tqdm(loader):
        if Y == None:
            continue
        model.eval()

        input_ids = X[0].to(device=device)
        attention_mask = X[1].to(device=device)
        Y = Y.to(device=device)

        y_pred = model(input_ids=input_ids,
                       attention_mask=attention_mask).logits

        loss = loss_criterion(
            y_pred.view(-1, args.category), Y.view(-1).long())
        val_loss.append(loss.detach().cpu().numpy())

        y_pred = y_pred.detach().cpu().numpy().reshape(-1, args.category)
        y_prob = y_pred[:, -1]
        y_pred = y_pred.argmax(axis=-1)
        y_label = Y.detach().cpu().numpy().reshape(-1)

        y_preds.append(y_pred)
        y_labels.append(y_label)
        y_probs.append(y_prob)

    # calculate the metric
    prediction = np.concatenate(y_preds, axis=0)
    label = np.concatenate(y_labels, axis=0)
    probs = np.concatenate(y_probs, axis=0)
    print(len(label))

    precision, recall, f1, auc, clf_report = evaluateModel(
        prediction, label)
    mean_loss = sum(val_loss)/len(val_loss)

    return mean_loss, [precision, recall, f1, auc, clf_report]


def evaluateModel(prediction, label):

    if args.category == 3:
        target_names = ['1', '2', '3']

    micro_precision = precision_score(label, prediction, average='micro')
    macro_precision = precision_score(label, prediction, average='macro')
    macro_recall = recall_score(label, prediction, average='macro')
    macro_f1 = f1_score(label, prediction, average='macro')
    clf_report = classification_report(
        label, prediction, target_names=target_names)

    return micro_precision, macro_precision, macro_recall, macro_f1, clf_report


if __name__ == '__main__':

    class_num = args.category
    model_name = args.model_name

    loader_train, loader_val, loader_test = load_data(
        Batch_Size=args.batch_size, model_name=model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=class_num)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_criterion = nn.CrossEntropyLoss()

    min_f1 = -1

    for epoch in range(args.epochs):

        st_time = time.time()
        '''training'''
        print('training....', epoch)
        loss_train = train_eopch(loader_train)

        '''validating'''
        with torch.no_grad():
            print('validating......')
            val_loss, val_evaluation = test_epoch(loader_val)

        '''testing'''
        with torch.no_grad():
            print('testing......')
            test_loss, test_evaluation = test_epoch(loader_test)

        if val_evaluation[2] > min_f1:
            min_f1 = val_evaluation[2]
            best_epoch = epoch + 1
            best_val = val_evaluation
            best_test = test_evaluation
            best_loss = val_loss


        print("Epoch: {}".format(epoch+1))
        logger.info("Epoch: {}".format(epoch+1))
        print("Train loss: {}".format(loss_train))
        logger.info("Train loss: {}".format(loss_train))


        print("Best Epoch: {}".format(best_epoch))
        logger.info("Best Epoch: {}".format(best_epoch))

        print('Best val micro_precision:', best_val[0])
        print('Best val macro_precision:', best_val[1])
        print('Best val macro_recall:', best_val[2])
        print('Best val macro_f1:', best_val[3])
        print('Best val clf_report:\n', best_val[4])

        logger.info('Best val micro_precision: ' + str(best_val[0]))
        logger.info('Best val macro_precision: ' + str(best_val[1]))
        logger.info('Best val macro_recall: ' + str(best_val[2]))
        logger.info('Best val macro_f1: ' + str(best_val[3]))
        logger.info('Best val clf_report:\n ' + str(best_val[4]))

        print('Best test micro_precision:', best_test[0])
        print('Best test macro_precision:', best_test[1])
        print('Best test macro_recall:', best_test[2])
        print('Best test macro_f1:', best_test[3])
        print('Best test clf_report:\n', best_test[4])

        logger.info('Best test micro_precision: ' + str(best_test[0]))
        logger.info('Best test macro_precision: ' + str(best_test[1]))
        logger.info('Best test macro_recall: ' + str(best_test[2]))
        logger.info('Best test macro_f1: ' + str(best_test[3]))
        logger.info('Best test clf_report:\n ' + str(best_test[4]))

        print('time: {:.4f}s'.format(time.time() - st_time))
        logger.info('time: {:.4f}s\n'.format(time.time() - st_time))

        if(epoch+1 - best_epoch >= args.patience):

            sys.exit(0)
