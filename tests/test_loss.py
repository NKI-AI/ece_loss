# coding=utf-8
# Copyright (c) Andreas Panteli, Jonas Teuwen

"""Test the loss function facility classes."""

import torch
from torch.nn.modules.loss import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss

from ece_loss.loss import BECELoss, BECEWithLogitsLoss, ExclusiveCrossEntropyLoss


def _identity_mapping_single_input(data):
    """Used for disabling components of ECE loss (during debugging/testing)"""
    return data


def _identity_mapping_double_input(_, label):
    """Used for disabling components of ECE loss (during debugging/testing)"""
    return label


def _identity_mapping_single_input_torch_ones(data):
    """Used for disabling components of ECE loss (during debugging/testing)"""
    return torch.ones(len(data))


def _identity_mapping_single_input_torch_zeros(data):
    """Used for disabling components of ECE loss (during debugging/testing)"""
    return torch.zeros(len(data))


def test_ce_output():  # pylint: disable=R0914
    """Test the outpout of the ECE if the exclusivity condition is diables, i.e. == CE"""
    ece_losses = [BECELoss, BECEWithLogitsLoss, ExclusiveCrossEntropyLoss]
    ce_losses = [BCELoss, BCEWithLogitsLoss, CrossEntropyLoss]

    linear_input_0_1 = torch.Tensor(range(11)) / 10
    linear_input_minus1_1 = (torch.Tensor(range(11)) - 5) / 5
    linear_multiclass_input = torch.cat((linear_input_0_1.unsqueeze(1), linear_input_minus1_1.unsqueeze(1)), dim=1)
    inputs = [linear_input_0_1, linear_input_minus1_1, linear_multiclass_input]
    targets = [
        torch.zeros(len(inputs[0])).float(),
        torch.zeros(len(inputs[1])).float(),
        torch.zeros(len(inputs[2])).long(),
    ]

    identity_mapping = _identity_mapping_double_input
    for ece_loss, ce_loss, input_sample, target in zip(ece_losses, ce_losses, inputs, targets):
        ce_loss_from_ece = ece_loss(
            exclusivity_threshold_annealing=identity_mapping,
            exclusivity_threshold=1,
            background_sampling_annealing=identity_mapping,
            background_sampling_threshold=1,
            focal_loss_parameters=(1, 0),
        )
        ce_loss_from_ece.sample_from_background = _identity_mapping_single_input_torch_ones
        ce_loss_no_reduction = ce_loss(reduction="none")

        ce_out = ce_loss_no_reduction(input_sample, target)
        ece_out = ce_loss_from_ece(input_sample, target)
        if not torch.allclose(ce_out, ece_out):
            print(ce_out, ece_out)
        assert torch.allclose(ce_out, ece_out)


def test_input_output():
    """Given specific input the output should be as pre-defined here"""
    linear_input_0_1 = torch.Tensor(range(11)) / 10
    linear_input_minus1_1 = (torch.Tensor(range(11)) - 5) / 5
    linear_multiclass_input = torch.cat((linear_input_0_1.unsqueeze(1), linear_input_minus1_1.unsqueeze(1)), dim=1)
    ece_losses = [BECELoss, BECEWithLogitsLoss, ExclusiveCrossEntropyLoss]
    inputs = [linear_input_0_1, linear_input_minus1_1, linear_multiclass_input]
    expected_outputs = [
        [0.0000, 0.0167, 0.0380, 0.0632, 0.0932, 0.1293, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0549, 0.0660, 0.0789, 0.0937, 0.1105, 0.1293, 0.1504, 0.1735, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0860, 0.0937, 0.1018, 0.1105, 0.1196, 0.1293],
    ]
    for input_sample, expected_output, ece_loss in zip(inputs, expected_outputs, ece_losses):
        input_sample.requires_grad = True
        expected_output = torch.tensor(expected_output)

        loss = ece_loss()
        loss.set_epoch(300)
        loss.sample_from_background = _identity_mapping_single_input_torch_zeros  # all background (fully un-annotated)

        output = loss(input_sample).detach()
        if not torch.allclose(expected_output, output, atol=1e-04):
            print(expected_output, output)
        assert torch.allclose(expected_output, output, atol=1e-04)


def test_beceloss():
    """Basic functionality for BECE"""
    loss = BECELoss()
    batch_size = 32
    epoch = 100
    input_sample = torch.rand(batch_size, requires_grad=True)
    output = loss(input_sample, epoch=epoch)
    assert output.shape == input_sample.shape
    assert not output.isnan().any()


def test_becewithlogitsloss():
    """Basic functionality for BECEWithLogits"""
    loss = BECEWithLogitsLoss()
    batch_size = 32
    epoch = 100
    input_sample = torch.rand(batch_size, requires_grad=True)
    output = loss(input_sample, epoch=epoch)
    assert output.shape == input_sample.shape
    assert not output.isnan().any()


def test_exclusivecrossentropyloss():
    """Basic functionality for ECE"""
    loss = ExclusiveCrossEntropyLoss()
    batch_size = 32
    epoch = 100
    num_classes = 10
    input_sample = torch.randn(batch_size, num_classes, requires_grad=True)
    target = torch.empty(batch_size, dtype=torch.long).random_(num_classes)
    output = loss(input_sample, target, epoch)
    assert output.shape == target.shape
    assert not output.isnan().any()

    output = loss(input_sample, epoch=epoch)
    assert not output.isnan().any()


def test_example_beceloss():
    """The docstring example of BECE"""
    sigmoid = torch.nn.Sigmoid()
    loss = BECELoss()
    input_sample = torch.randn(3, requires_grad=True)
    target = torch.empty(3).random_(2)
    output = loss(sigmoid(input_sample), target)
    assert output.shape == target.shape
    assert not output.isnan().any()


def test_example_becelosswithlogits():
    """The docstring example of BECEWithLogits"""
    loss = BECEWithLogitsLoss()
    batch_size = torch.randint(32, (1, 1)).squeeze()
    input_sample = torch.rand(batch_size, requires_grad=True)
    output = loss(input_sample)
    assert output.shape == input_sample.shape
    assert not output.isnan().any()
    output = loss(input_sample).mean()


def test_example_exclusivecrossentropy():
    """The docstring example of ECE"""
    loss = ExclusiveCrossEntropyLoss()
    epoch = torch.randint(300, (1, 1)).squeeze()
    batch_size = torch.randint(32, (1, 1)).squeeze()
    num_classes = torch.randint(100, (1, 1)).squeeze() + 1
    input_sample = torch.randn(batch_size, num_classes, requires_grad=True)
    target = torch.empty(batch_size, dtype=torch.long).random_(num_classes)
    output = loss(input_sample, target, epoch)
    assert output.shape == target.shape
    assert not output.isnan().any()
    output = torch.mean(output)


def test_device():  # pylint: disable=R0914
    """The device inputs are handled corrects"""
    batch_size = 32
    num_classes = 10

    losses = [BECELoss, BECEWithLogitsLoss, ExclusiveCrossEntropyLoss]
    input_sizes = [(batch_size,), (batch_size,), (batch_size, num_classes)]
    target_sizes = [(batch_size,), (batch_size,), (batch_size,)]
    target_dtypes = [torch.float, torch.float, torch.long]
    devices = ["cpu", "gpu"]
    for input_size, target_size, loss_, target_dtype in zip(input_sizes, target_sizes, losses, target_dtypes):
        for device in devices:
            loss = loss_()
            predictions = torch.rand(input_size, requires_grad=True)
            target = torch.zeros(target_size, dtype=target_dtype)
            if device == "cpu":
                predictions = predictions.cpu()
                target = target.cpu()
            elif device == "gpu":
                if not torch.cuda.is_available():
                    print("No CUDA available, skip GPU testing.")
                    continue
                predictions = predictions.cuda()
                target = target.cuda()
            else:
                raise NotImplementedError

            output = loss(predictions)
            assert not output.isnan().any()
            output = loss(predictions, target)
            assert not output.isnan().any()
