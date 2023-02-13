# -*- coding: utf-8 -*-


from turtle import forward
import torch
from utils import *
from transformers import AutoModelForSequenceClassification


class StatisticsModule(torch.nn.Module):
    def __init__(self,
                 device,
                 num_statistics=6,
                 out_features=6
                 ) -> None:
        super(StatisticsModule, self).__init__()
        self.device = device

        self.fc = torch.nn.Linear(
            in_features=num_statistics,
            out_features=out_features
        )

    def forward(self, content_stat_vec):
        content_stat_vec = torch.Tensor(
            content_stat_vec).to(device=self.device)
        logits = self.fc(content_stat_vec)

        return logits


class LiwcModule(torch.nn.Module):
    def __init__(self,
                 device,
                 in_features=73,
                 out_features=20) -> None:
        super(LiwcModule, self).__init__()
        self.device = device

        self.fc = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features
        )

    def forward(self, liwc_count_vec):
        liwc_count_vec = torch.Tensor(
            liwc_count_vec).to(device=self.device)
        logits = self.fc(liwc_count_vec)

        return logits


class ContentStatConcat(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        model_name,
        device,
        num_statistics=6,
        num_liwc_features=73,
        content_feature_dim=10,
        statistics_feature_dim=3,
        liwc_feature_dim=20,
        fusion_output_size=10,
        dropout_p=0
    ):
        super(ContentStatConcat, self).__init__()
        self.device = device

        self.content_module = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=content_feature_dim)
        self.content_module = self.content_module.to(device)

        self.stat_module = StatisticsModule(
            device=device, num_statistics=num_statistics, out_features=statistics_feature_dim)
        self.stat_module = self.stat_module.to(device)

        self.liwc_module = LiwcModule(
            device=device, in_features=num_liwc_features, out_features=liwc_feature_dim)
        self.stat_module = self.stat_module.to(device)

        self.fusion = torch.nn.Linear(
            in_features=(content_feature_dim +
                         statistics_feature_dim + liwc_feature_dim),
            out_features=fusion_output_size
        )
        self.fc = torch.nn.Linear(
            in_features=fusion_output_size,
            out_features=num_classes
        )
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, content, content_stat_vec, liwc_count_vec):
        content_input_ids = content[0].to(device=self.device)
        content_attention_mask = content[1].to(device=self.device)

        content_features = torch.nn.functional.relu(
            self.content_module(input_ids=content_input_ids,
                                attention_mask=content_attention_mask).logits
        )

        stat_features = torch.nn.functional.relu(
            self.stat_module(content_stat_vec)
        )

        liwc_features = torch.nn.functional.relu(
            self.liwc_module(liwc_count_vec)
        )

        combined = torch.cat(
            [content_features, stat_features, liwc_features], dim=1
        )
        fused = self.dropout(
            torch.nn.functional.relu(
                self.fusion(combined)
            )
        )
        logits = self.fc(fused)
        pred = torch.nn.functional.softmax(logits, dim=-1)
        return pred, logits
