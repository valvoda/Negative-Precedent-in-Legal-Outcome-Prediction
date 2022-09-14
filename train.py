from model import BertClassifier
from logger import Logger

from transformers import AdamW, get_linear_schedule_with_warmup, BertModel, AutoModel, LongformerModel
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

import random
import time
import argparse
import pickle

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


class Classifier:

    def __init__(self, model, out_dim=2, epochs=4, learning_rate=3e-5,
                 dropout=0.2, n_hidden=50, batch_size=16, max_len=512,
                 arguments=False, architecture="classifier", use_claims=False,
                 model_name='bert', dataset_name='dataset', inference=False, claim_path=None, pos_path=None):
        self.device = self.set_cuda()

        if inference:
            print("Loaded trained pos model!")
            self.model = torch.load(claim_path, map_location=self.device)
            print("Loaded trained claim model!")
            self.claim_model = torch.load(pos_path, map_location=self.device)
        else:
            self.model = BertClassifier(model, out_dim, dropout=dropout, n_hidden=int(n_hidden),
                                        device=self.device, use_claims=use_claims, architecture=architecture)
            self.claim_model = BertClassifier(model, out_dim, dropout=dropout, n_hidden=int(n_hidden),
                                        device=self.device, use_claims=use_claims, architecture=architecture)

        self.optimizer = AdamW(self.model.parameters(),
                          lr=learning_rate, # lr=5e-5,    # Default learning rate
                          eps=1e-8    # Default epsilon value
                          )

        self.architecture = architecture
        self.epochs = epochs
        self.arguments = arguments
        self.batch_size = batch_size
        self.max_len = max_len
        self.n_labels = None

        # Set up the learning rate scheduler
        self.scheduler = None
        if self.architecture == "claim_outcome":
            # NOTE Tiago Experiments
            self.loss_fn = nn.BCELoss(reduction='none')
        elif self.architecture == "joint_model":
            # logits no softmax
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        # MULTI GPU support:
        if self.device == "cuda" and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        self.claim_model.to(self.device)
        self.set_seed()
        self.log = Logger(self.architecture, model_name, dropout, learning_rate, batch_size, n_hidden, arguments, dataset_name)

    def set_cuda(self):
        if torch.cuda.is_available():
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            return torch.device("cuda")
        else:
            print('No GPU available, using the CPU instead.')
            return torch.device("cpu")

    def set_seed(self, seed_value=42):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    def run_epoch(self, dataloader, epoch_i, eval=False):

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0
        all_truths, all_preds = [], []

        # Put the model into the training/validation mode
        if eval:
            self.model.eval()
            self.claim_model.eval()
        else:
            self.model.train()
            self.claim_model.train()

        # For each batch of training data...
        for step, batch in enumerate(dataloader):
            batch_counts += 1
            # Load batch to GPU

            b_input_ids, b_attn_mask, b_labels, b_claims = tuple(t.to(self.device) for t in batch)

            # Zero out any previously calculated gradients
            self.model.zero_grad()

            # Perform a forward pass. This will return logits.
            b_input_ids = b_input_ids.squeeze(1)
            b_attn_mask = b_attn_mask.squeeze(1)

            global_attention_mask = torch.zeros(b_input_ids.shape, dtype=torch.long, device=self.device)
            global_attention_mask[:, [0]] = 1


            if self.architecture == 'claim_outcome':
                pos_logits = self.model(input_ids=b_input_ids, attention_mask=b_attn_mask,
                                                      global_attention=global_attention_mask, claims=b_claims)

                claim_logits = self.claim_model(input_ids=b_input_ids, attention_mask=b_attn_mask,
                                        global_attention=global_attention_mask, claims=b_claims)

            elif self.architecture == 'joint_model':
                logits = self.model(input_ids=b_input_ids, attention_mask=b_attn_mask,
                                    global_attention=global_attention_mask, claims=b_claims)
                logits = logits.reshape(b_input_ids.shape[0], -1, 3)
            else:
                logits = self.model(input_ids=b_input_ids, attention_mask=b_attn_mask,
                                    global_attention=global_attention_mask, claims=b_claims)

            # Compute loss and accumulate the loss values
            if self.architecture == "mtl":
                loss1 = self.loss_fn(logits[:, :logits.size(1) // 2], b_labels[:, :b_labels.size(1) // 2].float())
                loss2 = self.loss_fn(logits[:, logits.size(1) // 2:], b_labels[:, b_labels.size(1) // 2:].float())
                loss1 = torch.mean(loss1)
                loss2 = torch.mean(loss2)
                loss = (loss1 + loss2) / 2
            elif self.architecture == "claim_outcome":

                # Training
                # claim = p(c | f)
                # positive = p(o+ | c=1, f)
                # Inference
                # p(o+ | f) = p(o+ | c=1, f) p(c | f) '+ p(o+ | c=0,f) p(c=0|f)'
                # p(o- | f) = (1 - p(o+ | c=1, f)) p(c | f)

                pos_labels = b_labels[:, :b_labels.size(1) // 2]

                if eval:
                    claims = torch.sigmoid(claim_logits)
                    pos = torch.sigmoid(pos_logits) * claims
                    neg = (1 - (torch.sigmoid(pos_logits))) * claims
                    not_claims = 1 - torch.sigmoid(claim_logits)
                    precedent = torch.cat((pos, neg), dim=1)

                    logits = torch.cat((not_claims.unsqueeze(-1), pos.unsqueeze(-1), neg.unsqueeze(-1)), dim=2)
                    D_out = int(b_labels.shape[1] / 2)
                    y = torch.zeros(b_labels.shape[0], D_out).long().to(self.device)
                    y[b_labels[:, :D_out].bool()] = 1
                    y[b_labels[:, D_out:].bool()] = 2
                    b_labels = y

                    pos_train = torch.sigmoid(pos_logits)
                    loss1 = self.loss_fn(pos_train, pos_labels.float())
                    loss1[b_claims == 0] = 0
                    loss2 = self.loss_fn(claims, b_claims.float())
                    loss1 = torch.mean(loss1)
                    loss2 = torch.mean(loss2)
                    loss = (loss1 + loss2) / 2
                else:
                    print("Pretrain claim prediction model and positive outcome prediction model separately!")
                    # claims = torch.sigmoid(claim_logits)
                    # pos = torch.sigmoid(pos_logits)
                    # # for logging purposes
                    # precedent = torch.cat((pos, claims), dim=1)
                    #
                    # loss1 = self.loss_fn(pos, pos_labels.float())
                    # loss1[b_claims == 0] = 0
                    # loss2 = self.loss_fn(claims, b_claims.float())
                    # loss1 = torch.mean(loss1)
                    # loss2 = torch.mean(loss2)
                    # loss = (loss1 + loss2) / 2

            elif self.architecture == "baseline_positive":
                loss = self.loss_fn(logits, b_labels[:, :b_labels.size(1) // 2].float())
                loss = torch.mean(loss)
                b_labels = b_labels[:, :b_labels.size(1) // 2]
            elif self.architecture == "baseline_negative":
                loss = self.loss_fn(logits, b_labels[:, b_labels.size(1) // 2:].float())
                loss = torch.mean(loss)
                b_labels = b_labels[:, b_labels.size(1) // 2:]
            elif self.architecture == "claims":
                loss = self.loss_fn(logits, b_claims.float())
                loss = torch.mean(loss)
                b_labels = b_claims
            else:
                D_out = int(b_labels.shape[1] / 2)
                y = torch.zeros(b_labels.shape[0], D_out).long().to(self.device)
                y[b_labels[:, :D_out].bool()] = 1
                y[b_labels[:, D_out:].bool()] = 2

                b_labels = y

                loss = self.loss_fn(logits.reshape(-1, 3), y.reshape(-1))
                loss = torch.mean(loss)

            batch_loss += loss.detach().item()
            total_loss += loss.detach().item()


            if self.architecture == 'joint_model' or self.architecture == 'claim_outcome':
                preds = logits.argmax(-1)
                all_truths += b_labels.cpu().float().tolist()
                all_preds += preds.cpu().float().tolist()
                pos_f1 = f1_score(np.array(all_truths) == 1,
                                  np.array(all_preds) == 1,
                                  average='micro') * 100
                neg_f1 = f1_score(np.array(all_truths) == 2,
                                  np.array(all_preds) == 2,
                                  average='micro') * 100
                o_f1 = f1_score(np.array(all_truths) == 0,
                                  np.array(all_preds) == 0,
                                  average='micro') * 100
                accuracy = (np.array(all_truths) == np.array(all_preds)).mean()
                # print(neg_f1, pos_f1, o_f1, accuracy)
            else:
                preds = torch.round(torch.sigmoid(logits))

                all_truths += b_labels.cpu().float().tolist()
                all_preds += preds.cpu().float().tolist()

            if not eval:
                # Perform a backward pass to calculate gradients
                loss.backward()
                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and the learning rate
                self.optimizer.step()
                self.scheduler.step()
                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    print(
                        f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {'-':^9} | {'-':^9} | {'-':^9} | {time_elapsed:^9.2f}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

        if self.architecture == 'joint_model' or self.architecture == 'claim_outcome':
            accuracy = (np.array(all_truths)==np.array(all_preds)).mean() * 100
            f1 = f1_score(np.array(all_truths) == 1, np.array(all_preds) == 1, average="micro") * 100
            recall = recall_score(np.array(all_truths) == 1, np.array(all_preds) == 1, average="micro") * 100
            precision = precision_score(np.array(all_truths) == 1, np.array(all_preds) == 1, average="micro") * 100

            o_f1 = f1_score(np.array(all_truths) == 0, np.array(all_preds) == 0, average="micro") * 100
            o_recall = recall_score(np.array(all_truths) == 0, np.array(all_preds) == 0, average="micro") * 100
            o_precision = precision_score(np.array(all_truths) == 0, np.array(all_preds) == 0, average="micro") * 100
        else:
            accuracy = accuracy_score(all_truths, all_preds) * 100
            f1 = f1_score(all_truths, all_preds, average="micro") * 100
            precision = precision_score(all_truths, all_preds, average="micro") * 100
            recall = recall_score(all_truths, all_preds, average="micro") * 100
        total_loss = total_loss / len(dataloader)

        if self.architecture == 'baseline_negative' or self.architecture == 'baseline_positive' or self.architecture == 'claims' or self.architecture == 'mtl':
            neg_accuracy = 0
            neg_f1 = 0
            neg_precision = 0
            neg_recall = 0
            o_f1, o_recall, o_precision = 0, 0, 0

        elif self.architecture == 'joint_model' or self.architecture == 'claim_outcome':
            neg_precision = precision_score(np.array(all_truths) == 2,
                              np.array(all_preds) == 2,
                              average='micro') * 100
            neg_f1 = f1_score(np.array(all_truths) == 2,
                              np.array(all_preds) == 2,
                              average='micro') * 100
            neg_recall = recall_score(np.array(all_truths) == 2,
                              np.array(all_preds) == 2,
                              average='micro') * 100
            neg_accuracy = (np.array(all_truths)==np.array(all_preds)).mean()

        else:
            neg_accuracy = accuracy_score(np.array(all_truths)[:, self.n_labels//2:], np.array(all_preds)[:, self.n_labels//2:]) * 100
            neg_f1 = f1_score(np.array(all_truths)[:, self.n_labels//2:], np.array(all_preds)[:, self.n_labels//2:], average='micro') * 100
            neg_precision = precision_score(np.array(all_truths)[:, self.n_labels//2:], np.array(all_preds)[:, self.n_labels//2:], average='micro') * 100
            neg_recall = recall_score(np.array(all_truths)[:, self.n_labels//2:], np.array(all_preds)[:, self.n_labels//2:], average='micro') * 100

        return total_loss, f1, precision, recall, accuracy, all_truths, all_preds, neg_f1, neg_precision, neg_recall, neg_accuracy, o_f1, o_recall, o_precision

    def initialise_scheduler(self, train_dataloader):
        # Total number of training steps
        total_steps = len(train_dataloader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                        num_warmup_steps=0,  # Default value
                                        num_training_steps=total_steps)

    def train(self, train_dataloader, val_dataloader, test_dataloader, inference):
        """
        Train the BertClassifier model.
        """

        if not inference:
            print("Start training...\n")

            self.initialise_scheduler(train_dataloader)

            stop_cnt = 0
            best_loss = 100

            for epoch_i in range(self.epochs):
                # =======================================
                #               Training
                # =======================================
                # Print the header of the result table
                print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Val F1':^9} | {'Val Prec':^9} | {'Val Rec':^9} | {'Elapsed':^9}")
                print("-" *105)

                train_loss, f1, precision, recall, accuracy, _, _, _, _, _, _, _, _, _ = self.run_epoch(train_dataloader, epoch_i, eval=False)

                # run validation
                with torch.no_grad():
                    val_loss, val_f1, val_precision, val_recall, val_accuracy, _, _, neg_val_f1, neg_val_precision, neg_val_recall, neg_val_accuracy, _, _, _ = self.run_epoch(val_dataloader, epoch_i, eval=True)

                # Print performance over the entire training data
                print("-" * 105)
                print(
                    f"{epoch_i + 1:^7} | all | {train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | "
                    f"{val_f1:^9.2f} | {val_precision:^9.2f} | {val_recall:^9.2f} | ")
                print("-" * 105)
                print(f"{epoch_i + 1:^7} | neg | {train_loss:^12.6f} | {val_loss:^10.6f} | {neg_val_accuracy:^9.2f} | "
                    f"{neg_val_f1:^9.2f} | {neg_val_precision:^9.2f} | {neg_val_recall:^9.2f} | ")
                print("-" * 105)
                print("\n")

                if val_loss < best_loss:
                    best_loss = val_loss
                    stop_cnt = 0
                    self.log.save_model(self.model)
                else:
                    stop_cnt += 1
                    print(f"No Improvement! Stop cnt {stop_cnt}")

                if stop_cnt == 1:
                    print(f"Early Stopping at {stop_cnt}")
                    self.model = self.log.load_model()
                    break

            print("Training complete!")

            with torch.no_grad():
                val_loss, val_f1, val_precision, val_recall, val_accuracy, _, _, neg_val_f1, neg_val_precision, neg_val_recall, neg_val_accuracy, _, _, _  = self.run_epoch(val_dataloader, epoch_i, eval=True)

            print("-" * 105)
            print(
                f"{epoch_i + 1:^7} | pos | {train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | "
                f"{val_f1:^9.2f} | {val_precision:^9.2f} | {val_recall:^9.2f} | ")
            print("-" * 105)
            print(f"{epoch_i + 1:^7} | neg | {train_loss:^12.6f} | {val_loss:^10.6f} | {neg_val_accuracy:^9.2f} | "
                  f"{neg_val_f1:^9.2f} | {neg_val_precision:^9.2f} | {neg_val_recall:^9.2f} | ")
            print("-" * 105)
            print("\n")
        else:
            epoch_i = 0

        with torch.no_grad():
            test_loss, test_f1, test_precision, test_recall, test_accuracy, all_truths, all_preds, neg_test_f1, neg_test_precision, neg_test_recall, neg_test_accuracy, o_test_f1, o_test_recall, o_test_precision = self.run_epoch(test_dataloader, epoch_i, eval=True)

        # Print performance over the entire training data
        print(
            f"{'All Loss':^12} | {'Acc':^9} | {'F1':^9} | {'Precission':^9} | {'Recall':^9}")
        print("-" * 40)
        print(f"{test_loss:^12.6f} | {test_accuracy:^9.2f} | {test_f1:^9.2f} | {test_precision:^9.2f} | {test_recall:^9.2f}")
        print(
            f"{'Neg Loss':^12} | {'Acc':^9} | {'F1':^9} | {'Precission':^9} | {'Recall':^9}")
        print("-" * 40)
        print(
            f"{test_loss:^12.6f} | {neg_test_accuracy:^9.2f} | {neg_test_f1:^9.2f} | {neg_test_precision:^9.2f} | {neg_test_recall:^9.2f}")
        print(f"{'Outcome Loss Loss':^12} | {'Acc':^9} | {'F1':^9} | {'Precission':^9} | {'Recall':^9}")
        print("-" * 40)
        print(
            f"{test_loss:^12.6f} | {test_accuracy:^9.2f} | {o_test_f1:^9.2f} | {o_test_precision:^9.2f} | {o_test_recall:^9.2f}")

        if not inference:
            results = [val_loss, val_f1, val_precision, val_recall, val_accuracy, test_loss, test_f1, test_precision, test_recall, test_accuracy]
            outputs = [all_truths, all_preds]
            self.log.save_results(results, outputs)

    def make_loader(self, input, mask, labels, claims, train=True):
        labels = torch.tensor(labels)
        claims = torch.tensor(claims)
        data = TensorDataset(input, mask, labels, claims)
        if train:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
        return dataloader

    def run(self, tokenized_dir=None, test=False, inference=False):

        with open(tokenized_dir + "/tokenized_train.pkl", "rb") as f:
            train_facts, train_masks, train_arguments, \
            train_masks_arguments, train_ids, train_claims, train_outcomes, _ =  pickle.load(f)

        with open(tokenized_dir + "/tokenized_dev.pkl", "rb") as f:
            val_facts, val_masks, val_arguments, \
            val_masks_arguments, val_ids, val_claims, val_outcomes, _ = pickle.load(f)

        with open(tokenized_dir + "/tokenized_test.pkl", "rb") as f:
            test_facts, test_masks, test_arguments, \
            test_masks_arguments, test_ids, test_claims, test_outcomes, _ = pickle.load(f)

        if test: test_size = 3
        else: test_size = 100000

        if self.arguments:
            print("Arguments + Facts in training data")
            # train_inputs = torch.cat((train_arguments, train_facts), dim=0)
            # train_masks = torch.cat((train_masks_arguments, train_masks), dim=0)
            # train_claims = np.concatenate((train_claims, train_claims))
            # train_outcomes = np.concatenate((train_outcomes, train_outcomes))
            # or
            train_inputs = train_arguments
            train_masks = train_masks_arguments
            # -----
            val_inputs = val_facts
            val_masks = val_masks
            test_inputs = test_facts
            test_masks = test_masks
        else:
            print("Facts in training data")
            train_inputs = train_facts
            train_masks = train_masks
            val_inputs = val_facts
            val_masks = val_masks
            test_inputs = test_facts
            test_masks = test_masks

        train_inputs, train_masks = train_inputs[:test_size, :, :self.max_len], train_masks[:test_size, :, :self.max_len]
        val_inputs, val_masks = val_inputs[:test_size, :, :self.max_len], val_masks[:test_size, :, :self.max_len]
        test_inputs, test_masks = test_inputs[:test_size, :, :self.max_len], test_masks[:test_size, :, :self.max_len]

        neg_train_labels = train_claims[:test_size, :] - train_outcomes[:test_size, :]
        neg_val_labels = val_claims[:test_size, :] - val_outcomes[:test_size, :]
        neg_test_labels = test_claims[:test_size, :] - test_outcomes[:test_size, :]

        pos_train_labels = train_outcomes[:test_size, :]
        pos_val_labels = val_outcomes[:test_size, :]
        pos_test_labels = test_outcomes[:test_size, :]

        # ensure there are no -1 in data
        pos_val_labels[pos_val_labels < 0] = 0
        pos_train_labels[pos_train_labels < 0] = 0
        pos_test_labels[pos_test_labels < 0] = 0

        neg_val_labels[neg_val_labels < 0] = 0
        neg_train_labels[neg_train_labels < 0] = 0
        neg_test_labels[neg_test_labels < 0] = 0

        train_labels = np.concatenate((pos_train_labels, neg_train_labels), axis=1)
        val_labels = np.concatenate((pos_val_labels, neg_val_labels), axis=1)
        test_labels = np.concatenate((pos_test_labels, neg_test_labels), axis=1)

        claim_train_labels = train_claims[:test_size, :]
        claim_val_labels = val_claims[:test_size, :]
        claim_test_labels = test_claims[:test_size, :]

        self.n_labels = len(train_labels[1])

        # Create the DataLoader for our training set
        train_dataloader = self.make_loader(train_inputs, train_masks, train_labels, claim_train_labels, train=True)
        val_dataloader = self.make_loader(val_inputs, val_masks, val_labels, claim_val_labels, train=False)
        test_dataloader = self.make_loader(test_inputs, test_masks, test_labels, claim_test_labels, train=False)

        self.log.ids = test_ids
        self.train(train_dataloader, val_dataloader, test_dataloader, inference)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=512, required=False)
    parser.add_argument("--batch_size", type=int, default=16, required=False)
    parser.add_argument("--learning_rate", type=float, default=3e-5, required=False)
    parser.add_argument("--dropout", type=float, default=0.2, required=False)
    parser.add_argument("--n_hidden", type=float, default=50, required=False)
    parser.add_argument("--arguments", dest='arguments', action='store_true')
    parser.add_argument("--claims", dest='claims', action='store_true')
    parser.add_argument("--test", dest='test', action='store_true')
    parser.add_argument("--inference", dest='inference', action='store_true')
    parser.add_argument("--epochs", type=int, default=10, required=False)
    parser.add_argument("--model", type=str, default="bert", required=False)
    parser.add_argument("--dataset", type=str, default="precedent", required=False) # precedent, alleged
    parser.add_argument("--architecture", type=str, default="mtl", required=False) # mtl baseline_positive baseline_negative claims claim_outcome joint_model claims
    parser.add_argument("--input", type=str, default="facts", required=False) # arguments
    parser.add_argument("--pos_path", type=str, default="results/precedent/baseline_positive/bert/facts/1cf14ebeb3164d268890c1e9a7cc929b/model.pt", required=False)
    parser.add_argument("--claim_path", type=str, default="results/precedent/claims/bert/facts/9dec8c4189e14db29363183d1704cdc8/model.pt", required=False)

    args = parser.parse_args()
    print(args)

    if args.model == "bert":
        model = BertModel.from_pretrained('bert-base-uncased', gradient_checkpointing=True, return_dict=True)
    elif args.model == "legal_bert":
        model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased", return_dict=True)
    elif args.model == "longformer":
        model = LongformerModel.from_pretrained('allenai/longformer-base-4096', gradient_checkpointing=True, return_dict=True)
    else:
        print("Error: Unsupported Model")

    if args.dataset == "Chalkidis":
        output_dim = 17*2
    elif args.dataset == "Outcome":
        output_dim = 14*2
    else:
        print("Error: Unsupported Dataset")

    tokenized_dir = "ECHR/" + args.dataset + "/" + args.model

    cl = Classifier(model, out_dim=output_dim, epochs=args.epochs, learning_rate=args.learning_rate,
                    dropout=args.dropout, n_hidden=args.n_hidden, batch_size=args.batch_size,
                    max_len=args.max_length, arguments=args.arguments, architecture=args.architecture,
                    use_claims=args.claims, model_name=args.model, dataset_name=args.dataset, inference=args.inference,
                    claim_path=args.claim_path, pos_path=args.pos_path)
    cl.run(tokenized_dir=tokenized_dir, test=args.test, inference=args.inference)
