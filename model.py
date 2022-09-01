import torch
import torch.nn as nn

# Create the BertClassfier class
class BertClassifier(nn.Module):
    """
    Bert Model for Classification Tasks.
    """

    def __init__(self, model, out_dim=2, dropout=0.2, n_hidden=50, device='cpu', longformer=False, use_claims=False, architecture='classifier'):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, n_hidden, out_dim

        self.model = model

        self.device = device
        self.longformer = False
        if model.name_or_path == 'allenai/longformer-base-4096':
            self.longformer = True

        self.mtl = False
        self.discriminate = False
        self.use_claims = use_claims

        if architecture == 'mtl':
            self.mtl = True
            D_out = D_out//2
        elif architecture == 'claim_outcome':
            D_out = D_out//2
        elif architecture == 'baseline_positive' or architecture == 'baseline_negative' or architecture == 'claims' or architecture == 'joint_model':
            D_out = D_out//2

        # for claim embeddings
        vocab_size = 19
        self.embedding = nn.Embedding(vocab_size, D_in)

        # Instantiate an one-layer feed-forward classifier for main task
        if architecture == 'joint_model':
            self.classifier_positive = nn.Sequential(
                nn.Linear(D_in, H),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(H, D_out * 3)
            )
        else:
            self.classifier_positive = nn.Sequential(
                nn.Linear(D_in, H),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(H, D_out)
            )

        # Instantiate an one-layer feed-forward classifier for auxilary task
        self.classifier_aux = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, D_out)
        )

    def process_claims(self, claims, outputs):
        # Introduce claims
        # claims = BATCH_N x LABEL_N

        last_hidden_state_cls = outputs.last_hidden_state[:, 0, :]
        embedded = self.embedding(claims).to(self.device)
        # last_hidden_state_cls concatenated with claim embeddings

        all_batches = torch.zeros(embedded.size(0), embedded.size(2))
        for i in range(embedded.size(0)):
            all_batches[i, :] = torch.mean(
                torch.stack([last_hidden_state_cls[i, :], embedded[i][claims[0] != 0].mean(0)]), dim=0)

        last_hidden_state_cls = all_batches.to(self.device)

        return last_hidden_state_cls

    def forward(self, input_ids, attention_mask, global_attention, claims):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        if self.longformer:
            outputs = self.model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention)
        else:
            outputs = self.model(input_ids, attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs.last_hidden_state[:, 0, :]

        # Introduce claims as input
        if self.use_claims:
            last_hidden_state_cls = self.process_claims(claims, outputs)

        # Feed input to classifier to compute logits of pos_precedent
        logits = self.classifier_positive(last_hidden_state_cls)

        if self.mtl:
            # Feed input to classifier to compute logits of neg_precedent
            logits_aux = self.classifier_aux(last_hidden_state_cls)
            logits = torch.cat((logits, logits_aux), dim=1)
        elif self.discriminate:
            # Feed input to classifier to compute logits of claims
            logits_aux = self.classifier_aux(last_hidden_state_cls)
            return logits, logits_aux

        return logits