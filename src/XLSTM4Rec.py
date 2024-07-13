import torch
from torch import nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss

from xLSTM.xLSTM import xLSTM

class xLSTM4Rec(SequentialRecommender):
    """
    xLSTM4Rec is a sequential recommendation model that utilizes xLSTM for capturing sequential dependencies in user-item interactions.

    Args:
        config (dict): Configuration dictionary.
        dataset (Dataset): Dataset object.
    """

    def __init__(self, config, dataset):
        """
        Initializes the xLSTM4Rec model.

        Args:
            config (dict): Configuration dictionary containing model parameters.
            dataset (Dataset): Dataset object containing the training and testing data.
        """
        super(xLSTM4Rec, self).__init__(config, dataset)

        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)
        
        self.xlstm_layers = nn.ModuleList([
            xLSTM("m", torch.zeros(config["train_batch_size"], config["MAX_ITEM_LIST_LENGTH"], config["hidden_size"]).to('cuda'),
                  factor=config["xlstm_factor"], depth=config["xlstm_depth"])
        ])
        
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' is in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initializes the weights of the model.

        Args:
            module (nn.Module): The module whose weights need to be initialized.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        """
        Forward pass of the model.

        Args:
            item_seq (torch.Tensor): The input sequence of item IDs.
            item_seq_len (torch.Tensor): The length of the item sequences.

        Returns:
            torch.Tensor: The output sequence representation.
        """
        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)
        
        for layer in self.xlstm_layers:
            item_emb = layer(item_emb)
        
        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        """
        Calculates the loss for the given interaction.

        Args:
            interaction (dict): The interaction dictionary containing user-item interactions.

        Returns:
            torch.Tensor: The calculated loss.
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        """
        Predicts the score for the given interaction.

        Args:
            interaction (dict): The interaction dictionary containing user-item interactions.

        Returns:
            torch.Tensor: The predicted scores.
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        """
        Predicts the scores for all items for the given interaction.

        Args:
            interaction (dict): The interaction dictionary containing user-item interactions.

        Returns:
            torch.Tensor: The predicted scores for all items.
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores

